#!/usr/bin/env python3
import re
import argparse
from pathlib import Path

def replace_in_file(in_path, out_path, pattern, repl):
    text = Path(in_path).read_text()
    new_text = re.sub(pattern, repl, text)
    Path(out_path).write_text(new_text)

def main():
    parser = argparse.ArgumentParser(
        description="Find/replace filenames in a text file (e.g. URDF) using regex."
    )
    parser.add_argument("input_file", help="Input text file (URDF)")
    parser.add_argument("output_file", help="Output file with replacements")
    parser.add_argument("pattern", help="Regex pattern to match (Python syntax)")
    parser.add_argument("replacement", help="Replacement string (supports regex groups)")
    args = parser.parse_args()

    replace_in_file(args.input_file, args.output_file, args.pattern, args.replacement)

if __name__ == "__main__":
    main()
