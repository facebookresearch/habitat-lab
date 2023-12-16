#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
