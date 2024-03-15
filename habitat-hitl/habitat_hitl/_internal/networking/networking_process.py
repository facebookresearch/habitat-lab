#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import pprint
import signal
import ssl
from datetime import datetime, timedelta
from multiprocessing import Process
import time
from typing import Any, Dict, Generator, List, Optional

import aiohttp.web
import websockets
from websockets.client import ClientConnection
from websockets.server import WebSocketServer

from habitat_hitl._internal.networking.frequency_limiter import (
    FrequencyLimiter,
)
from habitat_hitl._internal.networking.interprocess_record import (
    InterprocessRecord,
    KeyframeAndMessages,
    Message,
)
from habitat_hitl._internal.networking.keyframe_utils import (
    get_empty_keyframe,
    get_user_keyframe,
    update_consolidated_keyframe,
    update_consolidated_messages,
)

# Boolean variable to indicate whether to use SSL
use_ssl = False

networking_process = None


def launch_networking_process(interprocess_record: InterprocessRecord) -> None:
    global networking_process

    networking_process = Process(
        target=networking_main, args=(interprocess_record,)
    )
    networking_process.start()


def terminate_networking_process() -> None:
    global networking_process
    if networking_process:
        networking_process.terminate()
        networking_process = None


def create_ssl_context() -> ssl.SSLContext:
    """
    Create an SSL context.

    This is not currently needed by the HITL framework, but we leave this here for reference.

    If an SSL context is needed, in some cases a self-signed key is sufficient, at least for testing. To generate self_signed.pem and private.key:

    1. Install openssl on your OS if necessary. I used conda to install.
    2. Generate private.key:
        openssl genpkey -algorithm RSA -out private.key -pkeyopt rsa_keygen_bits:2048
    3. Generate temp.csr:
        openssl req -new -key private.key -out temp.csr
    4. Generate self_signed.pem. There are several prompts for info like country and organization. You can press return to use defaults for all of these.
        openssl x509 -req -days 365 -in temp.csr -signkey private.key -out self_signed.pem -outform PEM
    """
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    certfile = "./self_signed.pem"
    keyfile = "./private.key"
    assert os.path.exists(certfile) and os.path.exists(keyfile)
    ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return ssl_context


class ClientEntry:
    websocket: ClientConnection
    connection_id: int
    connection_timestamp: datetime
    recent_connection_activity_timestamp: datetime
    waiting_for_client_ready: bool
    needs_consolidated_keyframe: bool

    def __init__(self, websocket: ClientConnection):
        self.websocket = websocket
        self.connection_id = id(websocket)
        self.connection_timestamp = datetime.now()
        self.recent_connection_activity_timestamp = self.connection_timestamp
        self.waiting_for_client_ready = True
        self.needs_consolidated_keyframe = True


class NetworkManager:
    """
    This class handles client connections, including sending a stream of keyframes and receiving a stream of client states.

    async check_keyframe_queue should be awaited. It runs continuously (an infinite loop with an asyncio.sleep(0.02) on each iteration). See networking_main_async. check_keyframe_queue's main job is to extract keyframes from the multiprocess queue and consolidate them into self._consolidated_keyframe. When NetworkManager has a client, check_keyframe_queue will also send each incremental keyframe to the client.

    handle_connection should be used with a websocket server. See networking_main_async. receive_client_states will run continuously. `async for message in websocket` runs until the websocket closes, essentially awaiting each incoming message.

    A network error (e.g. closed socket due to client disconnect) will result in an exception handled either in check_keyframe_queue or serve (the caller of receive_client_states). The error is handled identically in either case.
    """

    def __init__(self, interprocess_record: InterprocessRecord):
        self._connected_clients: Dict[
            int, ClientConnection
        ] = {}  # Dictionary to store connected clients

        self._interprocess_record = interprocess_record
        self._networking_config = interprocess_record._networking_config

        # Limit how many messages/sec we send. Note the current server implementation sends
        # messages "one at a time" (waiting for confirmation of receipt from the
        # remote client OS), so there is probably no risk of overwhelming the
        # network connection bandwidth even when sending at a high rate.
        max_send_rate = None  # 10  # or set None to not limit
        self._send_frequency_limiter = FrequencyLimiter(max_send_rate)

        consolidated_messages: List[Message] = []
        for _ in range(interprocess_record._networking_config.max_client_count):
            consolidated_messages.append({})
        self._consolidated_keyframe_and_messages = KeyframeAndMessages(get_empty_keyframe(), consolidated_messages)

        self._waiting_for_client_ready = False
        self._needs_consolidated_keyframe = False
        self._waiting_for_app_ready = False
        self._recent_connection_activity_timestamp: Optional[datetime] = None
        self._user_slots: Dict[int, ClientEntry] = {}

    def occupy_user_slot(self, websocket: ClientConnection) -> int:
        """
        Find the lowest available user_index and assigns the specified client to it.
        Returns the user index.
        """
        assert(len(self._user_slots) < self._networking_config.max_client_count)

        # TODO [0mdc/networking]: This data is coupled.
        self._connected_clients[id(websocket)] = websocket
        client = ClientEntry(websocket)

        user_index = 0
        while user_index in self._user_slots:
            user_index += 1
        self._user_slots[user_index] = client
        return user_index
        
    def free_user_slot(self, user_index: int) -> None:
        """
        Free a user_index slot, deleting its data and leaving it available for another incoming connection.
        TODO: Relation to handle_disconnection.
        """
        assert(len(self._user_slots) > 0)
        assert(user_index in self._user_slots)
        del self._user_slots[user_index]

    def update_consolidated_keyframes_and_messages(self,
                                                   consolidated_keyframes_and_messages: KeyframeAndMessages,
                                                   keyframes_and_messages: List[KeyframeAndMessages]) -> None:
        for inc_keyframe_and_messages in keyframes_and_messages:
            update_consolidated_keyframe(
                consolidated_keyframes_and_messages.keyframe, inc_keyframe_and_messages.keyframe
            )
            update_consolidated_messages(
                consolidated_keyframes_and_messages.messages, inc_keyframe_and_messages.messages
            )

    async def receive_client_states(self, websocket: ClientConnection) -> None:
        connection_id = id(websocket)
        async for message in websocket:
            self._recent_connection_activity_timestamp = datetime.now()
            try:
                # Parse the received message as a JSON object
                client_state = json.loads(message)

                client_state["connectionId"] = connection_id
                for user_index, client in self._user_slots.items():
                    if client.connection_id == connection_id:
                        client_state["userId"] = user_index
                # TODO: This can happen in some cases (right after disconnect?).
                if "userId" not in client_state:
                    print("Invalid client state!")
                    client_state["userId"] = 0

                self._interprocess_record.send_client_state_to_main_thread(
                    client_state
                )

            except json.JSONDecodeError:
                print("Received invalid JSON data from the client.")
            except Exception as e:
                print(f"Error processing received pose data: {e}")

    def is_okay_to_send_keyframes(self, user_index: int) -> bool:
        return (
            self.has_connection()
            and user_index in self._user_slots
            and not self._user_slots[user_index].waiting_for_client_ready
            and not self._waiting_for_app_ready
        )

    async def check_keyframe_queue(self) -> None:
        # this runs continuously even when there is no client connection
        while True:
            time_start_ns = time.time_ns()
            inc_keyframes_and_messages = self._interprocess_record.get_queued_keyframes()

            if len(inc_keyframes_and_messages) > 0:
                # consolidate all inc keyframes into one inc_keyframe
                tmp_con_keyframe = inc_keyframes_and_messages[0]
                if len(inc_keyframes_and_messages) > 1:
                    self.update_consolidated_keyframes_and_messages(tmp_con_keyframe, inc_keyframes_and_messages[1:])
                    inc_keyframes_and_messages = [tmp_con_keyframe]

                wrapper_json: Dict[int, Any] = {}
                for user_index in self._user_slots.keys():
                    if self.is_okay_to_send_keyframes(user_index):
                        for keyframe_and_messages_to_send in inc_keyframes_and_messages:
                            self.process_kick_and_app_ready_message(keyframe_and_messages_to_send.messages[user_index])
                            user_keyframes_to_send = [get_user_keyframe(keyframe_and_messages_to_send, user_index)]
                            # This client may be joining "late", after we've already simulated
                            # some frames. To handle this case, we send a consolidated keyframe as
                            # the very first keyframe for the new client. It captures all the
                            # previous incremental keyframes since the server started.
                            slot = self._user_slots[user_index]
                            if slot.needs_consolidated_keyframe:
                                user_keyframes_to_send.insert(0, get_user_keyframe(self._consolidated_keyframe_and_messages, user_index))
                                slot.needs_consolidated_keyframe = False
                        
                        # Convert keyframes to JSON string
                        wrapper_obj = {"keyframes": user_keyframes_to_send}
                        wrapper_json[user_index] = json.dumps(wrapper_obj)

                # after we've converted our keyframes to send to json, update
                # our consolidated keyframe
                self.update_consolidated_keyframes_and_messages(self._consolidated_keyframe_and_messages, inc_keyframes_and_messages)

                tasks = {}
                for user_index in self._user_slots.keys():
                    if self.is_okay_to_send_keyframes(user_index):
                        slot = self._user_slots[user_index]
                        tasks[user_index] = slot.websocket.send(wrapper_json[user_index])
                        slot.recent_connection_activity_timestamp = (
                            datetime.now()
                        )

                for user_index in self._user_slots.keys():
                    if self.is_okay_to_send_keyframes(user_index):
                        try:
                            # This will raise an exception if the connection is broken,
                            # e.g. if the server lost its network connection.
                            await tasks[user_index]
                        except Exception:
                            print(f"Error sending to client.")  # TODO: Improve error message.
                            self.handle_disconnect(slot.connection_id)

                # limit how often we send
                # await self._send_frequency_limiter.limit_frequency_async()

                time_end_ns = time.time_ns()
                elapsed_s = (time_end_ns - time_start_ns) / (10 ** 9)
                keyframe_send_sleep_time = 0.02
                wait_time = keyframe_send_sleep_time - min(elapsed_s, keyframe_send_sleep_time)
                await asyncio.sleep(wait_time)

    def process_kick_and_app_ready_message(self, message: Message) -> None:
        # for kickClient, we require the requester to include the connection_id. This ensures we don't kick the wrong client. E.g. the requester recently requested to kick an idle client, but NetworkManager already dropped that client and received a new client connection.
        #if "kickClient" in message:
        #    connection_id = message["kickClient"]
        #    if connection_id in self._connected_clients:
        #        print(f"Kicking client {connection_id}.")
        #        websocket = self._connected_clients[connection_id]
                # Don't await this; we want to keep checking keyframes. Beware this means the connection will remain alive for some time after this.
        #        asyncio.create_task(websocket.close())

        # See hitl_defaults.yaml wait_for_app_ready_signal and ClientMessageManager.signal_app_ready
        if (
            self._waiting_for_app_ready
            and self.has_connection()
            and "isAppReady" in message
            and message["isAppReady"]
        ):
            self._waiting_for_app_ready = False

    def has_connection(self) -> bool:
        return len(self._connected_clients) > 0
    
    def can_accept_connection(self) -> bool:
        return len(self._connected_clients) < self._networking_config.max_client_count

    def handle_disconnect(self, connection_id: int) -> None:
        """
        To be called after a websocket has closed. Don't call this to close the websocket.
        """
        if len(self._connected_clients) == 0:
            return
        assert connection_id in self._connected_clients
        websocket = self._connected_clients[connection_id]
        assert websocket.close_reason != None  # Assert that the socket is closed.
        print(f"Closed connection to client  {websocket.remote_address}")
        del self._connected_clients[connection_id]
        # Sloppy: Search for slot by connection ID
        # TODO [0mdc/networking]: Consolidate all these states.
        for user_index, client in self._user_slots.items():
            if client.connection_id == connection_id:
                self.free_user_slot(user_index)
                break

    def parse_connection_record(self, message: str) -> Any:
        connection_record = None
        if message == "client ready!":
            # legacy message format for initial client message
            connection_record = {"isClientReady": True}
        else:
            connection_record = json.loads(message)
            if "isClientReady" not in connection_record:
                raise ValueError(
                    "isClientReady key not found in initial client message."
                )
        return connection_record

    async def handle_connection(self, websocket: ClientConnection) -> None:
        # Kick clients after limit is reached.
        if not self.can_accept_connection():
            await websocket.close()
            return

        # Store the client connection object in the dictionary
        user_index = self.occupy_user_slot(websocket)
        user_slot = self._user_slots[user_index]
        user_slot.waiting_for_client_ready = True
        user_slot.needs_consolidated_keyframe = True

        # TODO: Should we wait_for_app_ready_signal when maximum client count is reached?
        self._waiting_for_app_ready = (
            self._networking_config.wait_for_app_ready_signal
        )

        print(f"Connection from client {websocket.remote_address} assigned to user_index {user_index}!")

        try:
            print("Waiting for connection record from client...")
            message = await websocket.recv()

            try:
                connection_record = self.parse_connection_record(message)
            except Exception:
                raise RuntimeError(
                    f"unexpected message from client: {message}"
                )
            print("Client is ready!")
            connection_record["connectionId"] = user_slot.connection_id
            connection_record["userId"] = user_index  # TODO: [0mdc/networking] - Consume this userId
            self._interprocess_record.send_connection_record_to_main_thread(
                connection_record
            )
            user_slot.waiting_for_client_ready = False
            # On disconnect, receive_client_states will either terminate normally
            # or raise an exception (this depends on how cleanly the client closes
            # the connection). We handle either case in the finally block below.
            await self.receive_client_states(websocket)

        except Exception:
            print(f"Error receiving from client.")  # TODO: Improve error message.
            pass
        finally:
            self.handle_disconnect(user_slot.connection_id)

    # Sloppy: Connection sends/receives seem to sometimes hang for several minutes, making the server unresponsive to new connections. Let's try to detect when this happens and close the connection. Unclear if this is actually helping. I believe the underlying cause was improper configuration of the AWS load balancer and this has probably since been fixed.
    async def check_close_broken_connection(self) -> None:
        while True:
            try:
                await asyncio.sleep(5)
                # print("check_close_broken_connection heartbeat")
                if self.has_connection():
                    current_time = datetime.now()
                    if (
                        current_time
                        - self._recent_connection_activity_timestamp
                        >= timedelta(seconds=10)
                    ):
                        for connection_id in self._connected_clients:
                            print(f"closing broken connection {connection_id}")
                            asyncio.create_task(
                                self._connected_clients[connection_id].close()
                            )
                            #TODO: Need to free slot.
                            #self.handle_disconnect(connection_id)
            except Exception:
                # print(f"recoverable error in check_close_broken_connection: {e}")
                pass


async def start_websocket_server(
    network_mgr: NetworkManager, networking_config
) -> WebSocketServer:
    global use_ssl
    network_mgr_lambda = lambda ws, path: network_mgr.handle_connection(ws)
    ssl_context = create_ssl_context() if use_ssl else None
    websocket_server = await websockets.serve(
        network_mgr_lambda, "0.0.0.0", networking_config.port, ssl=ssl_context
    )
    print(
        f"NetworkManager started on networking thread. Listening for client websocket connections on port {networking_config.port}..."
    )
    return websocket_server


async def start_http_availability_server(
    network_mgr: NetworkManager, networking_config
) -> aiohttp.web.AppRunner:
    # TODO [0mdc/multiplayer]: Include max client count here.
    async def http_handler(request):
        # return an HTTP code to indicate available or not
        code = (
            networking_config.http_availability_server.code_unavailable
            if network_mgr.has_connection()
            else networking_config.http_availability_server.code_available
        )
        # print(f"Returned availability HTTP code {code}")
        return aiohttp.web.Response(status=code)

    app = aiohttp.web.Application()
    app.router.add_get("/", http_handler)
    runner = aiohttp.web.AppRunner(
        app, access_log=None
    )  # access_log=None to silence log spam
    await runner.setup()
    site = aiohttp.web.TCPSite(
        runner, "0.0.0.0", networking_config.http_availability_server.port
    )
    await site.start()
    print(
        f"HTTP availability server started on networking thread. Listening for HTTP requests on port {networking_config.http_availability_server.port}..."
    )

    return runner


async def networking_main_async(
    interprocess_record: InterprocessRecord,
) -> None:
    networking_config = interprocess_record._networking_config
    assert networking_config.enable
    assert networking_config.max_client_count > 0

    network_mgr = NetworkManager(interprocess_record)

    # Start servers
    websocket_server = await start_websocket_server(
        network_mgr, networking_config
    )
    http_runner = (
        await start_http_availability_server(network_mgr, networking_config)
        if networking_config.http_availability_server.enable
        else None
    )

    # Define tasks (concurrent looping coroutines)
    tasks = []
    tasks.append(asyncio.create_task(network_mgr.check_keyframe_queue()))
    tasks.append(asyncio.create_task(network_mgr.check_close_broken_connection()))

    # Handle SIGTERM. We should get this signal when we do networking_process.terminate(). See terminate_networking_process.
    stop: asyncio.Future = asyncio.Future()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)
    loop.add_signal_handler(signal.SIGQUIT, stop.set_result, None)
    loop.add_signal_handler(signal.SIGINT , stop.set_result, None)

    # Run tasks.
    tasks = asyncio.gather(*tasks)

    # Wait for tasks to be completed or for cancellation (from SIGTERM).
    await stop
    print("Networking process terminating.")

    # Close servers.
    websocket_server.close()
    await websocket_server.wait_closed()

    if http_runner:
        await http_runner.cleanup()
    
    # Cancel running tasks.
    tasks.cancel()


def exception_handler(loop, context):
    pprint(context)

def networking_main(interprocess_record: InterprocessRecord) -> None:
    # Set up the event loop and run the main coroutine
    loop = asyncio.get_event_loop()
    #loop.set_exception_handler(exception_handler)
    loop.run_until_complete(networking_main_async(interprocess_record))
    loop.close()
    print("Networking process terminated.")
