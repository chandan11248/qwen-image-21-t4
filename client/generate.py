#!/usr/bin/env python3
"""Generate images from anywhere through the mini-UI tunnel.

Usage:
    python generate.py "a fox in a snowy forest at night" --url https://<tunnel> --size 768 --steps 20 --out fox.png

Polls the job until done, then saves the PNG. Needs only the stdlib.
"""

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request


def api(base, method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()


def main():
    ap = argparse.ArgumentParser(description="Remote image generation via mini-UI tunnel")
    ap.add_argument("prompt")
    ap.add_argument("--url", required=True, help="Tunnel URL, e.g. https://xxx.trycloudflare.com")
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--out", default="output.png")
    ap.add_argument("--timeout", type=int, default=1800)
    a = ap.parse_args()
    base = a.url.rstrip("/")

    _, body = api(base, "POST", "/api/generate",
                  {"prompt": a.prompt, "size": a.size, "steps": a.steps})
    job_id = json.loads(body)["job_id"]
    print(f"job: {job_id}  (size={a.size}, steps={a.steps})")

    t0 = time.time()
    while time.time() - t0 < a.timeout:
        time.sleep(10)
        _, body = api(base, "GET", f"/api/status?id={job_id}")
        st = json.loads(body)
        print(f"[{int(time.time() - t0)}s] {st['status']}")
        if st["status"] == "done":
            break
        if st["status"] == "error":
            sys.exit(f"FAILED: {st.get('error')}")
    else:
        sys.exit("timed out waiting for image")

    _, png = api(base, "GET", f"/api/image?id={job_id}")
    with open(a.out, "wb") as f:
        f.write(png)
    print(f"saved {len(png) / 1024:.0f} KB -> {a.out}")


if __name__ == "__main__":
    main()
