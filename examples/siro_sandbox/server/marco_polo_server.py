#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
