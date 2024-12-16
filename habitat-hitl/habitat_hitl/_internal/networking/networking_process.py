#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import signal
import ssl
import time
import traceback
from datetime import datetime
from multiprocessing import Process
from typing import Dict, List

import aiohttp.web
from websockets.server import WebSocketServer, WebSocketServerProtocol, serve

from habitat_hitl._internal.networking.frequency_limiter import (
    FrequencyLimiter,
)
from habitat_hitl._internal.networking.interprocess_record import (
    InterprocessRecord,
)
from habitat_hitl._internal.networking.keyframe_utils import (
    get_empty_keyframe,
    get_user_keyframe,
    update_consolidated_keyframe,
    update_consolidated_messages,
)
from habitat_hitl.core.hydra_utils import ConfigObject
from habitat_hitl.core.types import (
    ClientState,
    ConnectionRecord,
    DisconnectionRecord,
    Keyframe,
    KeyframeAndMessages,
    Message,
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


class Client:
    def __init__(self, socket: WebSocketServerProtocol):
        self.socket = socket
        self.connection_id = id(socket)
        self.connection_timestamp = datetime.now()
        self.recent_connection_activity_timestamp = self.connection_timestamp
        self.waiting_for_client_ready = True
        self.needs_consolidated_keyframe = True


class NetworkManager:
    """
    This class handles client connections, including sending a stream of keyframes and receiving a stream of client states.

    async send_keyframes should be awaited. It runs continuously (an infinite loop with an asyncio.sleep(0.02) on each iteration). See networking_main_async. send_keyframes's main job is to extract keyframes from the multiprocess queue and consolidate them into self._consolidated_keyframe. When NetworkManager has a client, send_keyframes will also send each incremental keyframe to the client.

    handle_connection should be used with a websocket server. See networking_main_async. receive_client_states will run continuously. `async for message in websocket` runs until the websocket closes, essentially awaiting each incoming message.

    A network error (e.g. closed socket due to client disconnect) will result in an exception handled either in send_keyframes or serve (the caller of receive_client_states). The error is handled identically in either case.
    """

    def __init__(self, interprocess_record: InterprocessRecord):
        self._connected_clients: Dict[
            int, WebSocketServerProtocol
        ] = {}  # Dictionary to store connected clients

        self._interprocess_record = interprocess_record
        self._networking_config = interprocess_record._networking_config
        self._max_client_count = self._networking_config.max_client_count
        self._user_slots: Dict[int, Client] = {}

        # Limit how many messages/sec we send. Note the current server implementation sends
        # messages "one at a time" (waiting for confirmation of receipt from the
        # remote client OS), so there is probably no risk of overwhelming the
        # network connection bandwidth even when sending at a high rate.
        max_send_rate = None  # 10  # or set None to not limit
        self._send_frequency_limiter = FrequencyLimiter(max_send_rate)
        self._waiting_for_app_ready = False

        consolidated_messages: List[Message] = [
            {} for _ in range(self._max_client_count)
        ]
        self._consolidated_keyframe_and_messages = KeyframeAndMessages(
            get_empty_keyframe(), consolidated_messages
        )

    def _occupy_user_slot(self, websocket: WebSocketServerProtocol) -> int:
        """
        Find the lowest available user_index and assigns the specified client to it.
        Returns the user index.
        """
        assert len(self._user_slots) < self._max_client_count

        # TODO This is not needed anymore.
        self._connected_clients[id(websocket)] = websocket
        client = Client(websocket)

        user_index = 0
        while user_index in self._user_slots:
            user_index += 1
        self._user_slots[user_index] = client

        # Remove user-specific messages.
        self._consolidated_keyframe_and_messages.messages[user_index].clear()

        return user_index

    def _free_user_slot(self, user_index: int) -> None:
        """
        Free a user_index slot, deleting its data and leaving it available for another incoming connection.
        """
        # TODO: Couple with handle_disconnection.
        assert len(self._user_slots) > 0
        assert user_index in self._user_slots
        del self._user_slots[user_index]

        # Remove user-specific messages.
        self._consolidated_keyframe_and_messages.messages[user_index].clear()

    def _update_consolidated_keyframes_and_messages(
        self,
        consolidated_keyframes_and_messages: KeyframeAndMessages,
        inc_keyframe_and_messages: KeyframeAndMessages,
    ) -> None:
        update_consolidated_keyframe(
            consolidated_keyframes_and_messages.keyframe,
            inc_keyframe_and_messages.keyframe,
        )
        update_consolidated_messages(
            consolidated_keyframes_and_messages.messages,
            inc_keyframe_and_messages.messages,
        )

    async def receive_client_states(
        self, websocket: WebSocketServerProtocol
    ) -> None:
        connection_id = id(websocket)
        async for message in websocket:
            try:
                # Parse the received message as a JSON object
                client_state: ClientState = json.loads(message)

                client_state["connectionId"] = connection_id

                for user_index, client in self._user_slots.items():
                    if client.connection_id == connection_id:
                        client_state["userIndex"] = user_index
                # TODO: This can happen in some cases (immediately after disconnect?).
                if "userIndex" not in client_state:
                    print("Invalid client state!")
                    client_state["userIndex"] = 0

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

    def _check_kick_client(self):
        kicked_user_indices = (
            self._interprocess_record.get_queued_kick_signals()
        )
        for user_index in kicked_user_indices:
            if user_index in self._user_slots:
                print(f"Kicking client {user_index}.")
                user_slot = self._user_slots[user_index]
                # Don't await this; we want to keep checking keyframes.
                # Beware that the connection will remain alive for some time after this.
                asyncio.create_task(user_slot.socket.close())

    async def send_keyframes(self) -> None:
        # this runs continuously even when there is no client connection
        user_json_strings: Dict[int, str] = {}
        while True:
            user_json_strings.clear()
            self._check_kick_client()

            time_start_ns = time.time_ns()
            inc_keyframes_and_messages = (
                self._interprocess_record.get_queued_keyframes()
            )

            if len(inc_keyframes_and_messages) > 0:
                tmp_con_keyframe = inc_keyframes_and_messages[0]

                # Discard messages for disconnected users.
                messages = tmp_con_keyframe.messages
                for user_index in range(len(messages)):
                    if user_index not in self._user_slots:
                        messages[user_index].clear()

                # Consolidate all inc keyframes into one inc_keyframe
                if len(inc_keyframes_and_messages) > 1:
                    for i in range(1, len(inc_keyframes_and_messages)):
                        self._update_consolidated_keyframes_and_messages(
                            tmp_con_keyframe, inc_keyframes_and_messages[i]
                        )
                    inc_keyframes_and_messages = [tmp_con_keyframe]

                for user_index in self._user_slots.keys():
                    slot = self._user_slots[user_index]
                    message = inc_keyframes_and_messages[0].messages[
                        user_index
                    ]

                    # See hitl_defaults.yaml wait_for_app_ready_signal and ClientMessageManager.signal_app_ready
                    if (
                        self._waiting_for_app_ready
                        and self.has_connection()
                        and "isAppReady" in message
                        and message["isAppReady"]
                    ):
                        self._waiting_for_app_ready = False

                    if self.is_okay_to_send_keyframes(user_index):
                        # This client may be joining "late", after we've already simulated
                        # some frames. To handle this case, we send a consolidated keyframe as
                        # the very first keyframe for the new client. It captures all the
                        # previous incremental keyframes since the server started.
                        user_keyframes_to_send: List[Keyframe] = []
                        if slot.needs_consolidated_keyframe:
                            user_keyframes_to_send.insert(
                                0,
                                get_user_keyframe(
                                    self._consolidated_keyframe_and_messages,
                                    user_index,
                                ),
                            )
                            slot.needs_consolidated_keyframe = False

                        # Create final user keyframes by combining keyframes and user messages.
                        for (
                            keyframe_and_messages_to_send
                        ) in inc_keyframes_and_messages:
                            user_keyframes_to_send.append(
                                get_user_keyframe(
                                    keyframe_and_messages_to_send, user_index
                                )
                            )

                        # Convert keyframes to JSON string
                        wrapper_obj = {"keyframes": user_keyframes_to_send}
                        user_json_strings[user_index] = json.dumps(wrapper_obj)

                # after we've converted our keyframes to send to json, update
                # our consolidated keyframe
                for inc_keyframe_and_messages in inc_keyframes_and_messages:
                    self._update_consolidated_keyframes_and_messages(
                        self._consolidated_keyframe_and_messages,
                        inc_keyframe_and_messages,
                    )

                tasks = {}
                for user_index in self._user_slots.keys():
                    if self.is_okay_to_send_keyframes(user_index):
                        slot = self._user_slots[user_index]
                        tasks[user_index] = slot.socket.send(
                            user_json_strings[user_index]
                        )
                        slot.recent_connection_activity_timestamp = (
                            datetime.now()
                        )

                for user_index in range(self._max_client_count):
                    if self.is_okay_to_send_keyframes(user_index):
                        slot = self._user_slots[user_index]
                        try:
                            # This will raise an exception if the connection is broken,
                            # e.g. if the server lost its network connection.
                            await tasks[user_index]
                        except Exception as e:
                            print(f"Error sending to client. Error: {e}.")
                            self.handle_disconnect(slot.connection_id)

            time_end_ns = time.time_ns()
            elapsed_s = (time_end_ns - time_start_ns) / (10**9)
            keyframe_send_sleep_time = 0.02
            wait_time = keyframe_send_sleep_time - min(
                elapsed_s, keyframe_send_sleep_time
            )
            await asyncio.sleep(wait_time)

    def has_connection(self) -> bool:
        """
        Returns true if at least one user is connected.
        """
        return len(self._connected_clients) > 0

    def is_server_available(self) -> bool:
        """
        Returns true if the server should advertise itself as available.
        """
        return (
            not self.has_connection()
            and self._interprocess_record.new_connections_allowed()
        )

    def can_accept_connection(self) -> bool:
        """
        Returns true if the server had the capacity for a new connection.
        """
        return (
            len(self._connected_clients)
            < self._networking_config.max_client_count
            and self._interprocess_record.new_connections_allowed()
        )

    def handle_disconnect(self, connection_id: int) -> None:
        """
        To be called after a websocket has closed. Don't call this to close the websocket.
        """
        if len(self._connected_clients) == 0:
            return

        # TODO: Connections can sometimes be closed multiple times.
        if connection_id not in self._connected_clients:
            print("Already disconnected.")
            return

        # Ensure that the connection is closed.
        websocket = self._connected_clients[connection_id]
        asyncio.create_task(websocket.close())

        print(f"Closed connection to client {connection_id}")
        del self._connected_clients[connection_id]

        # TODO: Consolidate these states.
        found = False
        for user_index, client in self._user_slots.items():
            if client.connection_id == connection_id:
                self._free_user_slot(user_index)
                found = True
                break

        assert found

        disconnection_record: DisconnectionRecord = {}
        disconnection_record["connectionId"] = connection_id
        disconnection_record["userIndex"] = user_index
        disconnection_record["timestamp"] = str(int(time.time()))
        self._interprocess_record.send_disconnection_record_to_main_thread(
            disconnection_record
        )

    def parse_connection_record(self, message: str) -> ConnectionRecord:
        connection_record: ConnectionRecord = json.loads(message)
        # Hack: connection record may be missing. Try to get it from client state.
        if (
            "isClientReady" not in connection_record
            and "connectionParamsDict" in connection_record
        ):
            connection_record = connection_record["connectionParamsDict"]
            if "isClientReady" not in connection_record:
                raise ValueError(
                    "isClientReady key not found in initial client message."
                )
        return connection_record

    async def handle_connection(
        self, websocket: WebSocketServerProtocol
    ) -> None:
        # Kick clients after limit is reached.
        if not self.can_accept_connection():
            await websocket.close()
            return

        # Store the client connection object in the dictionary
        user_index = self._occupy_user_slot(websocket)
        user_slot = self._user_slots[user_index]
        user_slot.waiting_for_client_ready = True
        user_slot.needs_consolidated_keyframe = True
        connection_id = user_slot.connection_id

        # TODO: Add _waiting_for_all_clients. Also add config to turn on this signal.
        self._waiting_for_app_ready = (
            self._networking_config.wait_for_app_ready_signal
        )

        print(
            f"Connection from client {connection_id} assigned to user_index {user_index}."
        )

        try:
            print("Waiting for connection record from client...")

            connection_record = None
            time_out = 10
            try:
                async for message in websocket:
                    if time_out <= 0:
                        raise (Exception("Timeout."))
                    try:
                        connection_record = self.parse_connection_record(
                            message
                        )
                        break
                    except Exception:
                        print(
                            f"Unable to get connection record from client message: {message}."
                        )
            except Exception:
                raise (
                    Exception(
                        "Client disconnected while sending connection record."
                    )
                )

            if connection_record is None:
                raise (Exception("Client did not send connection record."))

            print("Client is ready!")
            connection_record["connectionId"] = connection_id
            connection_record["userIndex"] = user_index
            connection_record["timestamp"] = str(int(time.time()))

            # Copy test connection parameters from "mock_connection_params_dict".
            if hasattr(self._networking_config, "mock_connection_params_dict"):
                mock_connection_params_dict = (
                    self._networking_config.mock_connection_params_dict
                )
                if mock_connection_params_dict is not None and isinstance(
                    mock_connection_params_dict, ConfigObject
                ):
                    for (
                        key,
                        value,
                    ) in mock_connection_params_dict.__dict__.items():
                        connection_record[key] = value

            self._interprocess_record.send_connection_record_to_main_thread(
                connection_record
            )
            user_slot.waiting_for_client_ready = False
            # On disconnect, receive_client_states will either terminate normally
            # or raise an exception (this depends on how cleanly the client closes
            # the connection). We handle either case in the finally block below.
            await self.receive_client_states(websocket)

        except Exception as e:
            print(f"Error receiving from client: {e}")
        finally:
            await websocket.close()
            self.handle_disconnect(connection_id)


async def start_websocket_server(
    network_mgr: NetworkManager, networking_config
) -> WebSocketServer:
    global use_ssl
    network_mgr_lambda = lambda ws, path: network_mgr.handle_connection(ws)
    ssl_context = create_ssl_context() if use_ssl else None
    websocket_server = await serve(
        network_mgr_lambda, "0.0.0.0", networking_config.port, ssl=ssl_context
    )
    print(
        f"NetworkManager started on networking thread. Listening for client websocket connections on port {networking_config.port}..."
    )
    return websocket_server


async def start_http_availability_server(
    network_mgr: NetworkManager, networking_config
) -> aiohttp.web.AppRunner:
    async def get_status(request):
        # return an HTTP code to indicate available or not
        code = (
            networking_config.http_availability_server.code_available
            if network_mgr.is_server_available()
            else networking_config.http_availability_server.code_unavailable
        )
        # print(f"Returned availability HTTP code {code}")
        return aiohttp.web.Response(status=code)

    async def get_server_state(request):
        return aiohttp.web.json_response(
            {
                "accepting_users": network_mgr.is_server_available(),
                "user_count": len(network_mgr._user_slots),
            },
            text=None,
            body=None,
            status=200,
            reason=None,
            headers=None,
            content_type="application/json",
            dumps=json.dumps,
        )

    app = aiohttp.web.Application()
    app.router.add_get("/", get_status)
    app.router.add_get("/status", get_server_state)
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

    # Start servers.
    websocket_server = await start_websocket_server(
        network_mgr, networking_config
    )
    http_runner = (
        await start_http_availability_server(network_mgr, networking_config)
        if networking_config.http_availability_server.enable
        else None
    )

    # Define tasks (concurrent looping coroutines).
    tasks: List[asyncio.Future] = []
    tasks.append(asyncio.create_task(network_mgr.send_keyframes()))

    # Handle termination signals.
    # We should get SIGTERM when we do networking_process.terminate(). See terminate_networking_process.
    stop: asyncio.Future = asyncio.Future()
    loop = asyncio.get_event_loop()
    stop_signals = [
        signal.SIGTERM,
        signal.SIGQUIT,
        signal.SIGINT,
        signal.SIGHUP,
    ]
    for stop_signal in stop_signals:
        loop.add_signal_handler(stop_signal, stop.set_result, None)
    # Add the stop signal as a task.
    tasks.append(stop)

    # Run tasks.
    abort = False
    while tasks:
        # Execute tasks until one is done (or fails).
        done_tasks, pending = await asyncio.wait(
            tasks, return_when=asyncio.FIRST_COMPLETED
        )
        for task in done_tasks:
            # Print exception for failed tasks.
            try:
                await task
            except Exception as e:
                print(f"Exception raised in network process. Aborting: {e}.")
                traceback.print_exc()
                abort = True
        # Abort if exception was raised, or if a termination signal was caught.
        if abort or stop.done():
            if stop.done():
                print(f"Caught termination signal: {stop.result()}.")
            break
        # Resume pending tasks.
        tasks = pending

    # Terminate network process.
    print("Networking process terminating...")

    # Close servers.
    websocket_server.close()
    await websocket_server.wait_closed()

    if http_runner:
        await http_runner.cleanup()


def networking_main(interprocess_record: InterprocessRecord) -> None:
    # Set up the event loop and run the main coroutine
    loop = asyncio.get_event_loop()
    loop.run_until_complete(networking_main_async(interprocess_record))
    loop.close()
    print("Networking process terminated.")
