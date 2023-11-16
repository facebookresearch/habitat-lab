import asyncio

import websockets


async def client():
    uri = "ws://192.168.4.58:8888"  # Replace with your server's IP and port

    async with websockets.connect(uri) as websocket:
        welcome_message = await websocket.recv()
        print(f"Received: {welcome_message}")

        while True:
            await websocket.send("marco")
            response = await websocket.recv()
            print(f"Sent 'marco', Received: {response}")
            await asyncio.sleep(1)


asyncio.get_event_loop().run_until_complete(client())
