#!/usr/bin/env python3
"""
OpenAI Python SDK test for the local Diffusers server.

Prereqs:
  pip install openai
Optionally set:
  export OPENAI_BASE_URL="http://127.0.0.1:8500/v1"
  export OPENAI_API_KEY="not-needed"   # our server doesn't enforce auth by default

Usage examples:
  python test_openai_sdk.py --prompt "a cozy cabin in the woods, watercolor" --size 256x256 --response-format url --output openai_sdk_url.png
  python test_openai_sdk.py --prompt "a watercolor fox" --response-format b64_json --output openai_sdk_b64.png
"""

import argparse
import base64
import os
import sys
import urllib.request


def save_bytes_to_file(data: bytes, path: str):
    with open(path, "wb") as f:
        f.write(data)


def main():
    parser = argparse.ArgumentParser(description="Test OpenAI SDK against local server")
    parser.add_argument("--prompt", default="a small red boat on a lake, minimalism")
    parser.add_argument("--size", default="256x256")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--response-format", default="url", choices=["url", "b64_json"], dest="response_format")
    parser.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--output", default="openai_sdk.png")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:8500/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "not-needed"))
    args = parser.parse_args()

    try:
        from openai import OpenAI
    except Exception as e:
        print("OpenAI library not installed. Install with `pip install openai`.")
        print("Error:", e)
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)

    # 1) Models list (optional sanity check)
    try:
        models = client.models.list()
        ids = [m.id for m in models.data]
        print("/v1/models ->", len(ids), "models")
    except Exception as e:
        print("Warning: models.list() failed:", e)

    # 2) Image generation
    try:
        result = client.images.generate(
            model=args.model,
            prompt=args.prompt,
            size=args.size,
            n=args.n,
            response_format=args.response_format,
        )
    except Exception as e:
        print("images.generate failed:", e)
        sys.exit(2)

    if not result or not getattr(result, "data", None):
        print("No data returned from images.generate")
        sys.exit(3)

    item = result.data[0]
    if args.response_format == "url":
        url = getattr(item, "url", None)
        if not url:
            print("No url in response item")
            sys.exit(4)
        print("image url:", url)
        with urllib.request.urlopen(url, timeout=30) as resp:
            content = resp.read()
            save_bytes_to_file(content, args.output)
        print("saved:", args.output)
    else:
        b64 = getattr(item, "b64_json", None)
        if not b64:
            print("No b64_json in response item")
            sys.exit(5)
        raw = base64.b64decode(b64)
        save_bytes_to_file(raw, args.output)
        print("saved:", args.output)


if __name__ == "__main__":
    main()
