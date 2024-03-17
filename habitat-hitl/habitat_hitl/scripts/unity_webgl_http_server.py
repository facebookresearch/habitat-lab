#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from http.server import HTTPServer, SimpleHTTPRequestHandler


class UnityRequestHandler(SimpleHTTPRequestHandler):
    """
    BaseHTTPRequestHandler for serving Unity WebGL builds.
    Sets the Content-Type and Content-Encoding headers required by browsers.
    """

    def end_headers(self):
        if self.path.endswith(".gz"):
            self.send_header("Content-Encoding", "gzip")
        super().end_headers()

    def do_GET(self):
        path = self.translate_path(self.path)
        if path.endswith(".js.gz"):
            with open(path, "rb") as file:
                content = file.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript")
                self.end_headers()
                self.wfile.write(content)
        elif path.endswith(".wasm.gz"):
            with open(path, "rb") as file:
                content = file.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/wasm")
                self.end_headers()
                self.wfile.write(content)
        elif path.endswith(".gz"):
            with open(path, "rb") as file:
                content = file.read()
                self.send_response(200)
                self.send_header("Content-Type", self.guess_type(path))
                self.end_headers()
                self.wfile.write(content)
        else:
            super().do_GET()


def start_server(path: str, hostname: str, port: int) -> None:
    """Start the server."""
    os.chdir(path)
    server = HTTPServer((hostname, port), UnityRequestHandler)
    print(f"Serving Unity build at: 'http://{hostname}:{port}'.")
    server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Unity WebGL HTTP Server",
        description=(
            """
            Simple HTTP server that serves Unity WebGL builds.
            Designed for local emulation of content provision services like S3.
            Unlike a normal HTTP server, it sets the Content-Type and Content-Encoding headers required by browsers.
            """
        ),
    )
    parser.add_argument(
        "--path",
        type=str,
        help="Path to the Unity WebGL build (where 'index.html' is located).",
    )
    parser.add_argument(
        "--hostname",
        type=str,
        default="localhost",
        help="Server hostname.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=3333,
        help="Server port.",
    )

    args = parser.parse_args()
    start_server(args.path, args.hostname, args.port)
