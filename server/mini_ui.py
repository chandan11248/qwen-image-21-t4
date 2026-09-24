#!/usr/bin/env python3
"""Qwen-Image-2.1 (GGUF) on Kaggle T4 — ChatGPT-style front end.

Tiny stdlib-only web server that sits in front of a local ComfyUI instance
(UnetLoaderGGUF + qwen_image CLIP + VAE) and exposes prompt -> image over HTTP,
so it stays usable even through slow tunnels.

Run next to ComfyUI (same machine):
    python mini_ui.py --port 8189

Then open http://<host>:8189 in any browser: text box + Generate button.
JSON API:
    POST /api/generate  {"prompt": str, "size": 768, "steps": 20} -> {"job_id": str}
    GET  /api/status?id=<job_id>    -> {"status": "running|done|error", "elapsed": s}
    GET  /api/image?id=<job_id>     -> image/png (when done)
"""

import argparse
import io
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HTML_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen T4 Image</title>
<style>body{font-family:sans-serif;max-width:640px;margin:24px auto;padding:0 12px;background:#111;color:#eee}
textarea{width:100%;box-sizing:border-box;background:#222;color:#eee;border:1px solid #444;border-radius:8px;padding:10px}
button{background:#4a7dff;color:#fff;border:0;border-radius:8px;padding:10px 22px;font-size:16px;margin-top:8px}
select,input{background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
img{max-width:100%;border-radius:8px;margin-top:12px}#st{margin-top:10px;color:#9db4ff}</style></head><body>
<h2>Qwen-Image-2.1 (T4)</h2>
<textarea id="p" rows="3" placeholder="Describe the image..."></textarea><br>
size <select id="s"><option value="512">512 test (~2 min)</option><option value="768" selected>768 fast (~4 min)</option><option value="1024">1024 best (~9 min)</option></select>
steps <input id="n" type="number" value="20" min="5" max="40" style="width:56px">
<button id="g">Generate</button>
<div id="st"></div><img id="img">
<script>
let timer=null;
g.onclick = async () => {
  clearInterval(timer); img.removeAttribute('src');
  st.textContent = 'sending prompt...';
  const r = await fetch('/api/generate', {method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({prompt: p.value, size: +s.value, steps: +n.value})});
  if (!r.ok) { st.textContent = 'error: ' + (await r.text()).slice(0,200); return; }
  const id = (await r.json()).job_id;
  st.textContent = 'queued... keep this page open (a few minutes)';
  timer = setInterval(async () => {
    const s2 = await (await fetch('/api/status?id=' + id)).json();
    st.textContent = 'status: ' + s2.status + ' (' + s2.elapsed + 's)';
    if (s2.status === 'done') { clearInterval(timer); img.src = '/api/image?id=' + id; st.textContent = 'done in ' + s2.elapsed + 's'; }
    if (s2.status === 'error') { clearInterval(timer); st.textContent = 'error: ' + s2.error; }
  }, 10000);
};
</script></body></html>"""


def build_workflow(prompt, diffusion, encoder, vae, size, steps, seed):
    """ComfyUI API-format workflow for the Qwen-Image-2.1 pipeline.

    Loader is picked by file type: .gguf -> UnetLoaderGGUF (ComfyUI-GGUF),
    .safetensors -> standard UNETLoader.
    """
    if diffusion.endswith(".gguf"):
        loader = {"class_type": "UnetLoaderGGUF",
                  "inputs": {"unet_name": diffusion}}
    else:
        loader = {"class_type": "UNETLoader",
                  "inputs": {"unet_name": diffusion, "weight_dtype": "default"}}
    return {
        "1": loader,
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": encoder, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": vae}},
        "4": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": prompt,
                         "negative_prompt": "", "resolution": size}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": size, "height": size, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0],
                         "negative": ["4", 1], "latent_image": ["5", 0],
                         "seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple",
                         "denoise": 1.0}},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0],
                         "filename_prefix": "qwen21_uncensored_t4"}},
    }


class MiniUI:
    def __init__(self, comfy_url, diffusion, encoder, vae):
        self.comfy_url = comfy_url.rstrip("/")
        self.diffusion = diffusion
        self.encoder = encoder
        self.vae = vae
        self.jobs = {}

    # ---- ComfyUI HTTP helpers ----
    def _post(self, path, payload):
        req = urllib.request.Request(
            self.comfy_url + path, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def _get(self, path, timeout=30):
        with urllib.request.urlopen(self.comfy_url + path,
                                    timeout=timeout) as r:
            return r.read()

    def generate_png(self, prompt, size, steps):
        seed = random.randint(0, 2 ** 31 - 1)
        wf = build_workflow(prompt, self.diffusion, self.encoder, self.vae,
                            size, steps, seed)
        try:
            prompt_id = self._post("/prompt", {"prompt": wf})["prompt_id"]
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected prompt: {e.read().decode()[:500]}")
        t0 = time.time()
        while time.time() - t0 < 3600:
            time.sleep(5)
            hist = json.loads(self._get(f"/history/{prompt_id}").decode())
            if prompt_id in hist and (
                    hist[prompt_id].get("status", {}).get("completed")
                    or "outputs" in hist[prompt_id]):
                outputs = hist[prompt_id]["outputs"]
                break
        else:
            raise TimeoutError("ComfyUI generation timed out")
        for out in outputs.values():
            for img in out.get("images", []):
                url = ("/view?filename=" + urllib.parse.quote(img["filename"])
                       + "&subfolder=" + urllib.parse.quote(img.get("subfolder", ""))
                       + "&type=" + img.get("type", "output"))
                return self._get(url, timeout=120)
        raise RuntimeError("ComfyUI returned no images")

    # ---- HTTP server ----
    def make_handler(self):
        ui = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, body, ctype="text/html"):
                if isinstance(body, str):
                    body = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/":
                    return self._send(200, HTML_PAGE)
                q = urllib.parse.urlparse(self.path)
                qs = urllib.parse.parse_qs(q.query)
                if q.path == "/api/status":
                    j = ui.jobs.get(qs.get("id", [""])[0])
                    if not j:
                        return self._send(404, "no job", "text/plain")
                    el = int(time.time() - j["t0"])
                    return self._send(200, json.dumps({
                        "status": j["status"], "elapsed": el,
                        "error": j.get("error", "")}), "application/json")
                if q.path == "/api/image":
                    j = ui.jobs.get(qs.get("id", [""])[0])
                    if not j or j["status"] != "done":
                        return self._send(404, "not ready", "text/plain")
                    return self._send(200, j["png"], "image/png")
                return self._send(404, "?", "text/plain")

            def do_POST(self):
                if self.path != "/api/generate":
                    return self._send(404, "?", "text/plain")
                try:
                    data = json.loads(self.rfile.read(
                        int(self.headers.get("Content-Length", 0))))
                except Exception:
                    return self._send(400, "bad json", "text/plain")
                prompt = str(data.get("prompt", "")).strip()[:2000]
                if not prompt:
                    return self._send(400, "empty prompt", "text/plain")
                size = min(max(int(data.get("size", 768) or 768), 256), 1536)
                size = size // 32 * 32
                steps = min(max(int(data.get("steps", 20) or 20), 5), 40)
                jid = uuid.uuid4().hex[:12]
                ui.jobs[jid] = {"status": "running", "png": None,
                                "error": "", "t0": time.time()}

                def run():
                    try:
                        png = ui.generate_png(prompt, size, steps)
                        ui.jobs[jid].update(status="done", png=png)
                    except Exception as e:  # noqa: BLE001
                        ui.jobs[jid].update(status="error",
                                            error=str(e)[:500])

                threading.Thread(target=run, daemon=True).start()
                return self._send(200, json.dumps({"job_id": jid}),
                                  "application/json")

        return H

    def serve(self, port):
        srv = ThreadingHTTPServer(("127.0.0.1", port), self.make_handler())
        print(f"Mini UI on http://127.0.0.1:{port}  (ComfyUI: {self.comfy_url})")
        srv.serve_forever()


def main():
    ap = argparse.ArgumentParser(description="Tiny front end for Qwen-Image GGUF on ComfyUI")
    ap.add_argument("--port", type=int, default=8189)
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--diffusion", default="qwen-image-2.1-UC-int8_convrot.safetensors")
    ap.add_argument("--encoder", default="qwen3vl_8b_int8_convrot.safetensors")
    ap.add_argument("--vae", default="qwen_image_2.1_vae_bf16.safetensors")
    a = ap.parse_args()
    MiniUI(a.comfy_url, a.diffusion, a.encoder, a.vae).serve(a.port)


if __name__ == "__main__":
    main()
