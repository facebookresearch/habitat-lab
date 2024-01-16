#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import ssl
import threading

import websockets

from habitat_hitl._internal.networking import multiprocessing_config
from habitat_hitl._internal.networking.frequency_limiter import (
    FrequencyLimiter,
)
from habitat_hitl._internal.networking.keyframe_utils import (
    get_empty_keyframe,
    update_consolidated_keyframe,
)

# Boolean variable to indicate whether to use SSL
use_ssl = False

networking_process = None
exit_event = None


def launch_networking_process(interprocess_record):
    # see multiprocessing_config to switch between real and dummy multiprocessing

    global networking_process
    global exit_event

    # multiprocessing.dummy.Process is actually a thread and requires special logic
    # to terminate it.
    exit_event = (
        threading.Event() if multiprocessing_config.use_dummy else None
    )

    networking_process = multiprocessing_config.Process(
        target=networking_main, args=(interprocess_record, exit_event)
    )
    networking_process.start()


def terminate_networking_process():
    global networking_process
    global exit_event
    if multiprocessing_config.use_dummy:
        if exit_event:
            exit_event.set()
            exit_event = None
    else:
        if networking_process:
            networking_process.terminate()
            networking_process = None


def create_ssl_context():
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


class NetworkManager:
    """
    This class handles client connections, including sending a stream of keyframes and
    receiving a stream of client states.

    When NetworkManager has no clients, async check_keyframe_queue runs continuously (an
    infinite loop with an asyncio.sleep(0.02) on each iteration). Its main job is
    to extract keyframes from the multiprocess queue and consolidate them into
    self._consolidated_keyframe.

    When NetworkManager has a client, check_keyframe_queue will also send each incremental
    keyframe to the client.

    Also when NetworkManager has a client, async receive_client_states will run continuously.
    `async for message in websocket` runs until the websocket closes, essentially
    awaiting each incoming message.

    These continuously-running async methods are run via the asyncio event loop (see
    networking_main).

    A network error (e.g. closed socket) will result in an exception handled either in
    check_keyframe_queue or serve (the caller of receive_client_states). The error is
    handled identically in either case.
    """

    def __init__(self, interprocess_record, exit_event):
        self._connected_clients = {}  # Dictionary to store connected clients

        self._interprocess_record = interprocess_record

        # Limit how many messages/sec we send. Note the current server implementation sends
        # messages "one at a time" (waiting for confirmation of receipt from the
        # remote client OS), so there is probably no risk of overwhelming the
        # network connection bandwidth even when sending at a high rate.
        max_send_rate = None  # 10  # or set None to not limit
        self._send_frequency_limiter = FrequencyLimiter(max_send_rate)

        self._exit_event = exit_event

        self._consolidated_keyframe = get_empty_keyframe()
        self._waiting_for_client_ready = False
        self._needs_consolidated_keyframe = False

    def update_consolidated_keyframes(self, keyframes):
        for inc_keyframe in keyframes:
            update_consolidated_keyframe(
                self._consolidated_keyframe, inc_keyframe
            )

    async def receive_client_states(self, websocket):
        async for message in websocket:
            try:
                # Parse the received message as a JSON object
                client_state = json.loads(message)

                self._interprocess_record.send_client_state_to_main_thread(
                    client_state
                )

            except json.JSONDecodeError:
                print("Received invalid JSON data from the client.")
            except Exception as e:
                print(f"Error processing received pose data: {e}")

    async def check_keyframe_queue(self):
        # this runs continuously even when there is no client connection
        while True:
            inc_keyframes = self._interprocess_record.get_queued_keyframes()

            if len(inc_keyframes):
                # consolidate all inc keyframes into one inc_keyframe
                tmp_con_keyframe = inc_keyframes[0]
                if len(inc_keyframes) > 1:
                    for i in range(1, len(inc_keyframes)):
                        update_consolidated_keyframe(
                            tmp_con_keyframe, inc_keyframes[i]
                        )
                    inc_keyframes = [tmp_con_keyframe]

                wrapper_json = None
                if (
                    self.has_connection()
                    and not self._waiting_for_client_ready
                ):
                    # This client may be joining "late", after we've already simulated
                    # some frames. To handle this case, we send a consolidated keyframe as
                    # the very first keyframe for the new client. It captures all the
                    # previous incremental keyframes since the server started.
                    keyframes_to_send = inc_keyframes
                    if self._needs_consolidated_keyframe:
                        keyframes_to_send = inc_keyframes.copy()
                        keyframes_to_send.insert(
                            0, self._consolidated_keyframe
                        )
                        self._needs_consolidated_keyframe = False

                    # Convert keyframes to JSON string
                    wrapper_obj = {"keyframes": keyframes_to_send}
                    wrapper_json = json.dumps(wrapper_obj)

                # after we've converted our keyframes to send to json, update
                # our consolidated keyframe
                self.update_consolidated_keyframes(inc_keyframes)

                if (
                    self.has_connection()
                    and not self._waiting_for_client_ready
                ):
                    websocket_id = list(self._connected_clients.keys())[0]
                    websocket = self._connected_clients[websocket_id]
                    try:
                        # This will raise an exception if the connection is broken,
                        # e.g. if the server lost its network connection.
                        await websocket.send(wrapper_json)
                    except Exception as e:
                        print(f"error sending to client: {e}")
                        self.handle_disconnect()

                # limit how often we send
                await self._send_frequency_limiter.limit_frequency_async()

            await asyncio.sleep(0.02)

    def has_connection(self):
        return len(self._connected_clients) > 0

    def handle_disconnect(self):
        if len(self._connected_clients) == 0:
            return
        assert len(self._connected_clients) == 1
        websocket_id = list(self._connected_clients.keys())[0]
        websocket = self._connected_clients[websocket_id]
        print(f"Closed connection to client  {websocket.remote_address}")
        del self._connected_clients[websocket_id]

    async def serve(self, websocket):
        # we only support one connected client at a time
        if self.has_connection():
            await websocket.close()
            return

        # Store the client connection object in the dictionary
        self._connected_clients[id(websocket)] = websocket
        self._waiting_for_client_ready = True
        self._needs_consolidated_keyframe = True

        print(f"Connection from client {websocket.remote_address}!")

        if self._exit_event and self._exit_event.is_set():
            await websocket.close()  # not sure if this is correct
            return

        try:
            print("Waiting for ready message from client...")
            message = await websocket.recv()

            if message == "client ready!":
                print("Client is ready!")
                self._waiting_for_client_ready = False
                # On disconnect, receive_client_states will either terminate normally
                # or raise an exception (this depends on how cleanly the client closes
                # the connection). We handle either case in the finally block below.
                await self.receive_client_states(websocket)
            else:
                raise RuntimeError(
                    f"unexpected message from client: {message}"
                )

        except Exception as e:
            print(f"error receiving from client: {e}")
        finally:
            self.handle_disconnect()


def networking_main(interprocess_record, exit_event):
    global use_ssl

    if multiprocessing_config.use_dummy:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    network_mgr = NetworkManager(interprocess_record, exit_event)
    network_mgr_lambda = lambda ws, path: network_mgr.serve(ws)
    ssl_context = create_ssl_context() if use_ssl else None
    start_server = websockets.serve(
        network_mgr_lambda, "0.0.0.0", 8888, ssl=ssl_context
    )

    check_keyframe_queue_task = asyncio.ensure_future(
        network_mgr.check_keyframe_queue()
    )

    asyncio.get_event_loop().run_until_complete(start_server)
    print("NetworkManager started on server thread. Waiting for clients...")
    while not (exit_event and exit_event.is_set()):
        asyncio.get_event_loop().run_until_complete(
            asyncio.sleep(1.0)
        )  # todo: investigate what sleep duration does here
    check_keyframe_queue_task.cancel()
    print("networking_main finished")
