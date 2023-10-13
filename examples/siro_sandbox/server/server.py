import asyncio
import json
import os
import ssl
import threading

import websockets

from . import multiprocessing_config
from .frequency_limiter import FrequencyLimiter
from .keyframe_utils import get_empty_keyframe, update_consolidated_keyframe

# Boolean variable to indicate whether to use SSL
use_ssl = False

server_process = None
exit_event = None


def launch_server_process(interprocess_record):
    # see multiprocessing_config to switch between real and dummy multiprocessing

    global server_process
    global exit_event

    # multiprocessing.dummy.Process is actually a thread and requires special logic
    # to terminate it.
    exit_event = (
        threading.Event() if multiprocessing_config.use_dummy else None
    )

    server_process = multiprocessing_config.Process(
        target=server_main, args=(interprocess_record, exit_event)
    )
    server_process.start()


def terminate_server_process():
    global server_process
    global exit_event
    if multiprocessing_config.use_dummy:
        if exit_event:
            exit_event.set()
            exit_event = None
    else:
        if server_process:
            server_process.terminate()
            server_process = None


def create_ssl_context():
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    certfile = "./examples/siro_sandbox/self_signed.pem"
    keyfile = "./examples/siro_sandbox/private.key"
    # see instructions for generating these files: https://docs.google.com/document/d/1hXuStZKNJafxLQVgl2zy2kyEf8Nke-bYogaPuHYT_M4/edit#bookmark=id.jva9nto0xpbe
    assert os.path.exists(certfile) and os.path.exists(keyfile)
    ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return ssl_context


class Server:
    def __init__(self, interprocess_record, exit_event):
        self._connected_clients = {}  # Dictionary to store connected clients

        self._interprocess_record = interprocess_record

        # Limit how many messages/sec we send. Note the current server implementation sends
        # messages "one at a time" (waiting for confirmation of receipt from the
        # remote client OS), so there is probably no risk of overwhelming the
        # network connection bandwidth even when sending at a high rate.
        max_send_rate = None  # or set None to not limit
        self._send_frequency_limiter = FrequencyLimiter(max_send_rate)

        self._exit_event = exit_event

        self._consolidated_keyframe = get_empty_keyframe()

    def update_consolidated_keyframes(self, keyframes):
        for inc_keyframe in keyframes:
            update_consolidated_keyframe(
                self._consolidated_keyframe, inc_keyframe
            )

    async def send_keyframes(self, websocket):
        needs_consolidated_keyframe = True

        while True:
            if self._exit_event and self._exit_event.is_set():
                break

            # todo: refactor to support N clients
            inc_keyframes = self._interprocess_record.get_queued_keyframes()

            if len(inc_keyframes):
                # This client may be joining "late", after we've already simulated
                # some frames. To handle this case, we send a consolidated keyframe as
                # the very first keyframe for the new client. It captures all the
                # previous incremental keyframes since the server started.
                keyframes_to_send = inc_keyframes
                if needs_consolidated_keyframe:
                    keyframes_to_send = inc_keyframes.copy()
                    keyframes_to_send.insert(0, self._consolidated_keyframe)
                    needs_consolidated_keyframe = False

                # Convert keyframes to JSON string
                wrapper_obj = {"keyframes": keyframes_to_send}
                wrapper_json = json.dumps(wrapper_obj)

                # note this awaits until the client OS has received the message
                await websocket.send(wrapper_json)

                self.update_consolidated_keyframes(inc_keyframes)

                # limit how often we send
                await self._send_frequency_limiter.limit_frequency_async()

            else:
                await asyncio.sleep(
                    0.02
                )  # todo: think about how best to do this

            # todo: don't send a message more often than 1.0 / maxSendRate

            # todo: don't busy loop here

    async def receive_avatar_pose(self, websocket):
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

    async def serve(self, websocket):
        # Store the client connection object in the dictionary
        self._connected_clients[id(websocket)] = websocket

        if self._exit_event and self._exit_event.is_set():
            await websocket.close()  # not sure if this is correct
            return

        try:
            print(f"waiting for message from client {id(websocket)}")
            message = await websocket.recv()

            if message == "client ready!":
                # await self.send_keyframes(websocket)

                # Start the tasks concurrently using asyncio.gather
                await asyncio.gather(
                    self.send_keyframes(websocket),
                    self.receive_avatar_pose(websocket),
                )

            else:
                raise RuntimeError(
                    f"unexpected message from client: {message}"
                )

        except Exception as e:
            # todo: not sure if we need to close websocket connection (most errors
            # correspond to a closed connection anyway, but maybe not all?)
            print(f"error serving client: {e}")
        finally:
            # Remove the client connection from the dictionary when it disconnects
            del self._connected_clients[id(websocket)]


def server_main(interprocess_record, exit_event):
    global use_ssl

    if multiprocessing_config.use_dummy:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    server_obj = Server(interprocess_record, exit_event)
    server_lambda = lambda ws, path: server_obj.serve(ws)
    ssl_context = create_ssl_context() if use_ssl else None
    start_server = websockets.serve(
        server_lambda, "0.0.0.0", 8888, ssl=ssl_context
    )

    asyncio.get_event_loop().run_until_complete(start_server)
    print("Server started on server thread. Waiting for clients...")
    while not (exit_event and exit_event.is_set()):
        asyncio.get_event_loop().run_until_complete(
            asyncio.sleep(1.0)
        )  # todo: investigate what sleep duration does here

    print("server_main finished")
