import asyncio
import websockets
import ssl
import json
import threading
import os

from .frequency_limiter import FrequencyLimiter
from .interprocess_record import interprocess_record, send_client_state_to_main_thread, get_queued_keyframes
from . import multiprocessing_config

# Boolean variable to indicate whether to use SSL
use_ssl = False

server_process = None
exit_event = None

def launch_server_process():
    # see multiprocessing_config to switch between real and dummy multiprocessing

    global server_process
    global exit_event

    # multiprocessing.dummy.Process is actually a thread and requires special logic
    # to terminate it.
    exit_event = threading.Event() if multiprocessing_config.use_dummy else None

    server_process = multiprocessing_config.Process(target=server_main, args=(interprocess_record, exit_event))
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
    certfile = './examples/siro_sandbox/self_signed.pem'
    keyfile='./examples/siro_sandbox/private.key'
    # see instructions for generating these files: https://docs.google.com/document/d/1hXuStZKNJafxLQVgl2zy2kyEf8Nke-bYogaPuHYT_M4/edit#bookmark=id.jva9nto0xpbe
    assert os.path.exists(certfile) and os.path.exists(keyfile)
    ssl_context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return ssl_context

class Server:

    def __init__(self, interprocess_record, exit_event):

        self._connected_clients = {}  # Dictionary to store connected clients

        # todo: get rid of this if not needed (we're currently using a global)
        self._interprocess_record = interprocess_record

        # Limit how many messages/sec we send. Note the current server implementation sends
        # messages "one at a time" (waiting for confirmation of receipt from the
        # remote client OS), so there is probably no risk of overwhelming the
        # network connection bandwidth even when sending at a high rate.
        max_send_rate = None  # or set None to not limit
        self._send_frequency_limiter = FrequencyLimiter(max_send_rate) 

        self._exit_event = exit_event

        self._hack_first_keyframe = None

    async def send_keyframes(self, websocket):

        did_send_hack_first_keyframe = False

        while True:

            if self._exit_event and self._exit_event.is_set():
                break

            # todo: refactor to support N clients
            keyframes = get_queued_keyframes()

            if len(keyframes):

                if not self._hack_first_keyframe:
                    self._hack_first_keyframe = keyframes[0]
                elif not did_send_hack_first_keyframe:
                    # hack: resend first keyframe to new clients
                    keyframes.insert(0, self._hack_first_keyframe)
                did_send_hack_first_keyframe = True

                # Convert keyframes to JSON string
                wrapper_obj = {"keyframes" : keyframes};
                wrapper_json = json.dumps(wrapper_obj)
                
                # note this awaits until the client OS has received the message
                await websocket.send(wrapper_json)
                # except Exception as e:
                #     print(f"Error sending keyframe data: {e}")
                #     break 

                # limit how often we send
                await self._send_frequency_limiter.limit_frequency_async()

            else:
                await asyncio.sleep(0.02)  # todo: think about how best to do this

            # todo: don't send a message more often than 1.0 / maxSendRate

            # todo: don't busy loop here



    async def receive_avatar_pose(self, websocket):
        async for message in websocket:
            try:
                # Parse the received message as a JSON object
                client_state = json.loads(message)

                send_client_state_to_main_thread(client_state)

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
                    self.receive_avatar_pose(websocket)
                )                

            else:
                raise RuntimeError(f"unexpected message from client: {message}")

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
    start_server = websockets.serve(server_lambda, "0.0.0.0", 8888, ssl=ssl_context)

    asyncio.get_event_loop().run_until_complete(start_server)
    print("Server started on server thread. Waiting for clients...")
    while not (exit_event and exit_event.is_set()):
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(1.0))  # todo: investigate what sleep duration does here

    print("server_main finished")