#!/usr/bin/env python3
import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from multiprocessing import Process


def start_server(host: str = "0.0.0.0", port: int = 8500):
    import importlib.util
    import pathlib
    import uvicorn

    path = pathlib.Path(__file__).parent / "serverasync.py"
    spec = importlib.util.spec_from_file_location("server_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["server_mod"] = mod
    spec.loader.exec_module(mod)
    uvicorn.run(mod.app, host=host, port=port, log_level="warning")


def wait_for_server(url: str, attempts: int = 50, delay: float = 0.3):
    for _ in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                return r.status == 200
        except Exception:
            time.sleep(delay)
    return False


def fetch_json(url: str, method: str = "GET", data: dict | None = None):
    if data is not None:
        enc = json.dumps(data).encode()
        req = urllib.request.Request(url, data=enc, headers={"Content-Type": "application/json"}, method=method)
    else:
        req = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(req, timeout=300) as r:
        body = r.read().decode()
        return r.status, json.loads(body)


def save_base64_png(b64: str, out_path: str):
    raw = base64.b64decode(b64)
    with open(out_path, "wb") as f:
        f.write(raw)


def main():
    parser = argparse.ArgumentParser(description="OpenAI-compatible endpoint smoke test")
    parser.add_argument("--prompt", default="a small red boat on a lake, minimalism")
    parser.add_argument("--size", default="256x256")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--response-format", default="url", choices=["url", "b64_json"], dest="response_format")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8500)
    parser.add_argument("--start-server", action="store_true", help="Start the server in a subprocess for the test")
    parser.add_argument("--output", default="openai_test.png")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"

    # Use SERVICE_URL so returned URLs are fetchable when bound to 0.0.0.0
    os.environ.setdefault("SERVICE_URL", base_url)

    p: Process | None = None
    try:
        if args.start_server:
            p = Process(target=start_server, kwargs={"host": "0.0.0.0", "port": args.port})
            p.start()
            ok = wait_for_server(f"{base_url}/api/status")
            if not ok:
                print("Server did not become ready in time", file=sys.stderr)
                sys.exit(1)

        # 1) list models
        status, models = fetch_json(f"{base_url}/v1/models")
        print("/v1/models:", status)
        if status != 200:
            print(models)
            sys.exit(2)

        # 2) generate
        payload = {
            "prompt": args.prompt,
            "size": args.size,
            "n": args.n,
            "response_format": args.response_format,
        }
        status, out = fetch_json(f"{base_url}/v1/images/generations", method="POST", data=payload)
        print("/v1/images/generations:", status)
        if status != 200:
            print(out)
            sys.exit(3)
        data = out.get("data", [])
        if not data:
            print("No data returned", file=sys.stderr)
            sys.exit(4)

        item = data[0]
        if args.response_format == "url":
            url = item.get("url")
            print("image url:", url)
            with urllib.request.urlopen(url, timeout=30) as img:
                content = img.read()
                with open(args.output, "wb") as f:
                    f.write(content)
            print("saved:", args.output)
        else:
            b64 = item.get("b64_json")
            if not b64:
                print("No b64_json returned", file=sys.stderr)
                sys.exit(5)
            save_base64_png(b64, args.output)
            print("saved:", args.output)

    finally:
        if p is not None:
            p.terminate()
            p.join(5)


if __name__ == "__main__":
    main()
