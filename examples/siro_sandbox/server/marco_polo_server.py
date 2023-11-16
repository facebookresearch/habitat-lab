import asyncio

import websockets


async def server(websocket, path):
    print("Client connected")

    await websocket.send("Welcome to the server!")

    try:
        while True:
            message = await websocket.recv()
            if message == "marco":
                await asyncio.sleep(1)  # Wait for 1 second
                await websocket.send("polo")
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


start_server = websockets.serve(server, "0.0.0.0", 8888)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
