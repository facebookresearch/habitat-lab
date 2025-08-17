#!/usr/bin/env python3
import json, sys

if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <file.json>")
    sys.exit(1)

path = sys.argv[1]

with open(path, "r") as f:
    data = json.load(f)

with open(path, "w") as f:
    # separators=(',', ':') removes all unnecessary whitespace
    json.dump(data, f, separators=(',', ':'))

print(f"Minified {path}")
