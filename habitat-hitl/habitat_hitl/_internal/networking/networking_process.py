#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import asyncio
import json
import os
import signal
import ssl

import aiohttp.web
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


def launch_networking_process(interprocess_record):
    # see multiprocessing_config to switch between real and dummy multiprocessing

    global networking_process

    # we don't support dummy multiprocessing at this time
    assert not multiprocessing_config.use_dummy

    networking_process = multiprocessing_config.Process(
        target=networking_main, args=(interprocess_record,)
    )
    networking_process.start()


def terminate_networking_process():
    global networking_process
    assert not multiprocessing_config.use_dummy
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
    This class handles client connections, including sending a stream of keyframes and receiving a stream of client states.

    async check_keyframe_queue should be awaited. It runs continuously (an infinite loop with an asyncio.sleep(0.02) on each iteration). See networking_main_async. check_keyframe_queue's main job is to extract keyframes from the multiprocess queue and consolidate them into self._consolidated_keyframe. When NetworkManager has a client, check_keyframe_queue will also send each incremental keyframe to the client.

    handle_connection should be used with a websocket server. See networking_main_async. receive_client_states will run continuously. `async for message in websocket` runs until the websocket closes, essentially awaiting each incoming message.

    A network error (e.g. closed socket due to client disconnect) will result in an exception handled either in check_keyframe_queue or serve (the caller of receive_client_states). The error is handled identically in either case.
    """

    def __init__(self, interprocess_record):
        self._connected_clients = {}  # Dictionary to store connected clients

        self._interprocess_record = interprocess_record

        # Limit how many messages/sec we send. Note the current server implementation sends
        # messages "one at a time" (waiting for confirmation of receipt from the
        # remote client OS), so there is probably no risk of overwhelming the
        # network connection bandwidth even when sending at a high rate.
        max_send_rate = None  # 10  # or set None to not limit
        self._send_frequency_limiter = FrequencyLimiter(max_send_rate)

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

    async def handle_connection(self, websocket):
        # we only support one connected client at a time
        if self.has_connection():
            await websocket.close()
            return

        # Store the client connection object in the dictionary
        self._connected_clients[id(websocket)] = websocket
        self._waiting_for_client_ready = True
        self._needs_consolidated_keyframe = True

        print(f"Connection from client {websocket.remote_address}!")

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


async def start_websocket_server(network_mgr, networking_config):
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


async def start_http_availability_server(network_mgr, networking_config):
    async def http_handler(request):
        # return an HTTP code to indicate available or not
        code = (
            networking_config.http_availability_server.code_unavailable
            if network_mgr.has_connection()
            else networking_config.http_availability_server.code_available
        )
        print(f"Returned availability HTTP code {code}")
        return aiohttp.web.Response(status=code)

    app = aiohttp.web.Application()
    app.router.add_get("/", http_handler)
    runner = aiohttp.web.AppRunner(app)
    await runner.setup()
    site = aiohttp.web.TCPSite(
        runner, "0.0.0.0", networking_config.http_availability_server.port
    )
    await site.start()
    print(
        f"HTTP availability server started on networking thread. Listening for HTTP requests on port {networking_config.http_availability_server.port}..."
    )

    return runner


async def networking_main_async(interprocess_record):
    networking_config = interprocess_record.networking_config
    assert networking_config.enable

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

    check_keyframe_queue_task = asyncio.ensure_future(
        network_mgr.check_keyframe_queue()
    )

    # Handle SIGTERM. We should get this signal when we do networking_process.terminate(). See terminate_networking_process.
    stop: asyncio.Future = asyncio.Future()
    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGTERM, stop.set_result, None)

    # This await essentially means "wait forever" (or until we get SIGTERM). Meanwhile, the other tasks we've started above (websocket server, http server, check_keyframe_queue_task) will also run forever in the asyncio event loop.
    await stop

    # Do cleanup code after we've received SIGTERM: close both servers and cancel check_keyframe_queue_task.
    websocket_server.close()
    await websocket_server.wait_closed()

    if http_runner:
        await http_runner.cleanup()

    check_keyframe_queue_task.cancel()


def networking_main(interprocess_record):
    # Set up the event loop and run the main coroutine
    loop = asyncio.get_event_loop()
    loop.run_until_complete(networking_main_async(interprocess_record))
    loop.close()
    print("networking_main finished")
