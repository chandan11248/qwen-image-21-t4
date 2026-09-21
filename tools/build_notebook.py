"""Builder for Kaggle T4 notebook: Qwen-Image-2.1-Uncensored-GGUF"""
import nbformat as nbf
import os

OUT_DIR = "/Users/owner/Desktop/alphaxiz/uncensored/kaggle_qwen21_uncensored_gguf"
os.makedirs(OUT_DIR, exist_ok=True)
NOTEBOOK_PATH = os.path.join(OUT_DIR, "qwen-image-2-1-uncensored-gguf-t4.ipynb")

nb = nbf.v4.new_notebook()
nb.metadata.update({
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.10"},
    "accelerator": "GPU",
})

# ---- Cell 0: markdown header ----
md0 = """# Qwen-Image-2.1 Uncensored (GGUF) — Kaggle T4 Notebook

Runs **[abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF)** on a Kaggle **T4 (16GB VRAM)** via **ComfyUI + leejet/ComfyUI-GGUF**.

**Model files used (T4-fit, from the model card):**
- Diffusion (GGUF): `qwen-image-2.1-Q4_K_M.gguf` — 4.60 GB (recommended balance, stays in VRAM)
- Text encoder: `text_encoders/qwen3vl_8b_int8_convrot.safetensors` — 9.35 GB (offloaded to CPU RAM, encoding runs once per prompt)
- VAE: `vae/qwen_image_2.1_vae_bf16.safetensors` — 676 MB

> Total download ≈ 14.6 GB. Q8_0 is **not** used (model card warns of `[136] vs [128]` shape-mismatch on some setups).

### Before you Run (Kaggle UI)
1. Right sidebar → **Accelerator** → select **GPU T4 x2** (or `GPU T4`).
2. Right sidebar → **Internet** → **ON** (needed to clone ComfyUI + download from Hugging Face).
3. **Run cells top-to-bottom.** First run takes ~10–20 min (ComfyUI clone + 14.6 GB download + server boot). Re-runs reuse `/kaggle/working` cache if you keep the session alive.
4. Output PNGs are saved to `/kaggle/working/` and displayed inline.

No safety checker / content filter in this GGUF release — output depends solely on your prompt and environment.

### Website mode (ChatGPT-easy, use it from any browser)
The notebook serves a **tiny page (type prompt → get image)** on port 8189 and exposes it via cloudflared.
Just open the printed `miniui` URL: text box + Generate button, nothing to build. A second tunnel exposes
ComfyUI itself (advanced/API use). For interactive use, set `KEEP_ALIVE_MINUTES` (e.g. 300) in the last cell.
URLs die when the session ends; each restart gives new URLs — don't share them publicly.

### Fast mode (default ON)
`FAST_MODE = True` generates at **768×768, 20 steps** (~2× faster than 1024/25). Set `FAST_MODE = False`
for full-quality 1024×1024, 25 steps.
"""

# ---- Cell 1: config + GPU check ----
code1 = """# ===== 0) Config — EDIT ME =====
REPO_ID = "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"

# Diffusion GGUF choice (model card: Q4_K_M recommended; Q4_0 smallest/fastest):
#   "qwen-image-2.1-Q4_K_M.gguf" (4.60GB) | "qwen-image-2.1-Q4_0.gguf" (4.05GB)
#   "qwen-image-2.1-Q5_K_M.gguf" (5.22GB) | "qwen-image-2.1-Q6_K.gguf" (5.88GB)
DIFFUSION_FILE = "qwen-image-2.1-Q4_K_M.gguf"

TEXT_ENCODER_FILE = "qwen3vl_8b_int8_convrot.safetensors"   # 9.35GB int8 — REQUIRED for T4, runs on CPU RAM
VAE_FILE = "qwen_image_2.1_vae_bf16.safetensors"            # 676MB

COMFY_PORT = 8188
COMFY_DIR = "/kaggle/working/ComfyUI"

# Generation defaults (T4-safe)
PROMPT = "cinematic portrait of a warrior queen at sunset, highly detailed, dramatic lighting, 35mm"
NEGATIVE_PROMPT = ""
WIDTH, HEIGHT = 1024, 1024   # full-quality size (used when FAST_MODE=False)
STEPS = 25
CFG = 1.0                    # keep 1.0 for official Qwen-Image-2.1 path
SAMPLER = "euler"
SCHEDULER = "simple"
SEED = 123456

# Fast preset: 768x768 + 20 steps ≈ 2x faster (≈4-5 min first image, ≈3-4 min after warmup)
FAST_MODE = True

# Website mode: expose ComfyUI via public cloudflared URL (no signup needed)
USE_TUNNEL = True
# Keep-alive loop minutes (holds the run + tunnel open; 0 = off so batch runs finish)
KEEP_ALIVE_MINUTES = 480  # 8h persistent tunnel service (burns ~8h GPU quota)

# Optional: HF token if repo ever goes gated (public now, leave empty)
HF_TOKEN = ""  # e.g. "hf_..."

print(f"REPO: {REPO_ID}")
print(f"Diffusion: {DIFFUSION_FILE} | TE: {TEXT_ENCODER_FILE} | VAE: {VAE_FILE}")
print(f"Target: {WIDTH}x{HEIGHT}, steps={STEPS}, cfg={CFG}, {SAMPLER}/{SCHEDULER}")
"""

code2 = """# ===== 1) Environment check (GPU / torch / disk) =====
import shutil, subprocess, sys

print("--- nvidia-smi ---")
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20).stdout[:3000])
except Exception as e:
    print(f"nvidia-smi failed: {e}\\nIf this shows no GPU, enable GPU T4 in Kaggle sidebar → Accelerator → GPU.")

import torch
print(f"torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"vram_total={torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

print("--- disk ---")
for p in ["/kaggle/working", "/tmp"]:
    try:
        u = shutil.disk_usage(p)
        print(f"{p}: total={u.total/1024**3:.1f}G used={u.used/1024**3:.1f}G free={u.free/1024**3:.1f}G")
    except Exception as e:
        print(p, e)

# Fail fast with a clear message if no GPU
assert torch.cuda.is_available(), "No CUDA GPU detected. In Kaggle: sidebar → Accelerator → GPU T4 x2, then re-run."
"""

code3 = """# ===== 2) Install ComfyUI + ComfyUI-GGUF (leejet fork, Qwen-Image 2.1 support) =====
import os, subprocess, sys

def run(cmd):
    print(f"$ {cmd}")
    r = subprocess.run(cmd, shell=True)
    assert r.returncode == 0, f"FAILED ({r.returncode}): {cmd}"
    return r

# Kaggle image already ships torch+CUDA — do NOT reinstall torch (slow + breaks CUDA).
# Only add the small missing deps.
run(f"{sys.executable} -m pip install -q --upgrade gguf huggingface_hub websocket-client requests")

if not os.path.isdir(COMFY_DIR):
    run(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI {COMFY_DIR}")
else:
    print(f"ComfyUI exists: {COMFY_DIR}, pulling latest (for Qwen-Image-2.1 nodes)...")
    run(f"git -C {COMFY_DIR} pull --ff-only || true")

# ComfyUI python deps (full install: Kaggle's torch 2.10+cu128 already satisfies torch pins,
# and full install pulls transitive deps like `trampoline` required by torchsde)
run(f"{sys.executable} -m pip install -q -r {COMFY_DIR}/requirements.txt || true")
# Belt-and-braces for the torchsde->trampoline import chain (missing this broke server boot on v1):
run(f"{sys.executable} -m pip install -q trampoline torchsde || true")
# Minimal runtime deps ComfyUI actually imports (safe, small):
run(f"{sys.executable} -m pip install -q pillow pyyaml tqdm einops safetensors aiohttp kornia spandrel soundfile av || true")

GGUF_NODE_DIR = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-GGUF")
if not os.path.isdir(GGUF_NODE_DIR):
    run(f"git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF {GGUF_NODE_DIR}")
else:
    run(f"git -C {GGUF_NODE_DIR} pull --ff-only || true")
run(f"{sys.executable} -m pip install -q -r {GGUF_NODE_DIR}/requirements.txt || true")

print("\\nComfyUI + ComfyUI-GGUF ready.")
print(f"ComfyUI: {COMFY_DIR}")
"""

code4 = """# ===== 3) Download weights (resume-safe, ~14.6 GB) =====
import os
from huggingface_hub import hf_hub_download

tok = HF_TOKEN.strip() or None
diff_dir = os.path.join(COMFY_DIR, "models", "diffusion_models")
te_dir   = os.path.join(COMFY_DIR, "models", "text_encoders")
vae_dir  = os.path.join(COMFY_DIR, "models", "vae")
unet_dir = os.path.join(COMFY_DIR, "models", "unet")  # legacy path some forks check
for d in [diff_dir, te_dir, vae_dir, unet_dir]:
    os.makedirs(d, exist_ok=True)

print("Downloading diffusion GGUF (4-6 GB)...")
p_diff = hf_hub_download(repo_id=REPO_ID, filename=DIFFUSION_FILE, local_dir=diff_dir,
                         token=tok, resume_download=True)
print("Downloading text encoder int8 (9.35 GB, slowest part)...")
p_te = hf_hub_download(repo_id=REPO_ID, filename=f"text_encoders/{TEXT_ENCODER_FILE}", local_dir=os.path.join(COMFY_DIR, "models"),
                       token=tok, resume_download=True)
print("Downloading VAE (676 MB)...")
p_vae = hf_hub_download(repo_id=REPO_ID, filename=f"vae/{VAE_FILE}", local_dir=os.path.join(COMFY_DIR, "models"),
                        token=tok, resume_download=True)

# Compat: some GGUF loaders look in models/unet — symlink/copy there (cheap hardlink attempt)
import shutil
legacy_gguf = os.path.join(unet_dir, DIFFUSION_FILE)
if not os.path.exists(legacy_gguf):
    try:
        os.link(p_diff, legacy_gguf)
    except Exception:
        shutil.copy(p_diff, legacy_gguf)

print("\\n--- files ---")
for p in [p_diff, p_te, p_vae]:
    sz = os.path.getsize(p)/1024**3
    print(f"{sz:5.2f} GB  {p}")

# Sanity: filenames the API workflow will request must exist
assert os.path.basename(p_diff) == DIFFUSION_FILE
assert os.path.exists(os.path.join(te_dir, TEXT_ENCODER_FILE)), f"missing TE: {TEXT_ENCODER_FILE}"
assert os.path.exists(os.path.join(vae_dir, VAE_FILE)), f"missing VAE: {VAE_FILE}"
print("\\nAll weights present.")
"""

code5 = """# ===== 4) Start ComfyUI server in background (lowvram for T4) =====
import subprocess, time, urllib.request, json, os, signal, atexit

COMFY_LOG = "/kaggle/working/comfyui.log"
BASE_URL = f"http://127.0.0.1:{COMFY_PORT}"

# Kill stale server on same port (re-runs)
try:
    subprocess.run(f"fuser -k {COMFY_PORT}/tcp 2>/dev/null || true", shell=True, timeout=10)
except Exception:
    pass
time.sleep(2)

logf = open(COMFY_LOG, "w")
proc = subprocess.Popen(
    ["python", "main.py", "--port", str(COMFY_PORT), "--lowvram",
     "--dont-print-server", "--disable-auto-launch", "--verbose", "INFO"],
    cwd=COMFY_DIR, stdout=logf, stderr=subprocess.STDOUT,
)
print(f"ComfyUI PID={proc.pid}, log={COMFY_LOG}")

def _stop():
    try:
        proc.terminate()
    except Exception:
        pass
atexit.register(_stop)

# Wait for server (up to ~3 min on first boot)
ok = False
for i in range(90):
    try:
        with urllib.request.urlopen(f"{BASE_URL}/system_stats", timeout=5) as r:
            info = json.loads(r.read().decode())
            print(f"ComfyUI up after {i*2}s. system={info.get('system', {})}")
            ok = True
            break
    except Exception:
        if i % 10 == 0:
            print(f"waiting... {i*2}s (see {COMFY_LOG})")
        time.sleep(2)

if not ok:
    print(f"--- tail {COMFY_LOG} ---")
    try:
        print(open(COMFY_LOG).read()[-6000:])
    except Exception as e:
        print(e)
    raise RuntimeError(f"ComfyUI did not start on :{COMFY_PORT}. Check {COMFY_LOG}.")

print(f"Server ready at {BASE_URL}")
"""

code6 = """# ===== 5) Generate via ComfyUI API (UnetLoaderGGUF + qwen_image CLIP + VAE) =====
import json, time, random, urllib.request, urllib.error, io
from PIL import Image
import matplotlib.pyplot as plt

BASE_URL = f"http://127.0.0.1:{COMFY_PORT}"

def build_workflow(prompt, diffusion_file, te_file, vae_file, w, h, steps, cfg, sampler, scheduler, seed):
    # Flat API workflow — avoids version-fragile subgraph templates.
    # Node classes: UnetLoaderGGUF (leejet/ComfyUI-GGUF), CLIPLoader(type=qwen_image),
    #   VAELoader, TextEncodeQwenImage21, EmptyLatentImage, KSampler, VAEDecode, SaveImage
    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": diffusion_file}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": te_file, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": vae_file}},
        "4": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": prompt,
                         "negative_prompt": NEGATIVE_PROMPT, "resolution": max(w, h)}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": w, "height": h, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": cfg,
                         "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": "qwen21_uncensored_t4"}},
    }

def queue_and_wait(workflow, timeout_s=1200, poll_s=3):
    data = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(f"{BASE_URL}/prompt", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:4000]
        raise RuntimeError(f"/prompt HTTP {e.code}: {body}")
    pid = resp["prompt_id"]
    print(f"queued prompt_id={pid}")
    t0 = time.time()
    while True:
        time.sleep(poll_s)
        with urllib.request.urlopen(f"{BASE_URL}/history/{pid}", timeout=30) as r:
            hist = json.loads(r.read().decode())
        if pid in hist:
            entry = hist[pid]
            if entry.get("status", {}).get("completed") or "outputs" in entry:
                return pid, entry
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"Timed out after {timeout_s}s waiting for {pid}")

def fetch_image(outputs):
    for node_id, out in outputs.items():
        for img in out.get("images", []):
            fn, sub, ftype = img["filename"], img.get("subfolder", ""), img.get("type", "output")
            url = f"{BASE_URL}/view?filename={urllib.parse.quote(fn)}&subfolder={urllib.parse.quote(sub)}&type={ftype}"
            with urllib.request.urlopen(url, timeout=60) as r:
                return Image.open(io.BytesIO(r.read())).convert("RGB"), fn
    raise RuntimeError(f"No images in outputs: {list(outputs.keys())}")

import urllib.parse

# Fast preset resolution/steps (full-quality fallback when FAST_MODE=False)
FW, FH, FSTEPS = (768, 768, 20) if FAST_MODE else (WIDTH, HEIGHT, STEPS)
print(f"mode={'FAST 768x768/20' if FAST_MODE else 'FULL 1024x1024/25'}")

seed = SEED if isinstance(SEED, int) else random.randint(0, 2**31 - 1)
wf = build_workflow(PROMPT, DIFFUSION_FILE, TEXT_ENCODER_FILE, VAE_FILE,
                    FW, FH, FSTEPS, CFG, SAMPLER, SCHEDULER, seed)
print(f"Generating {FW}x{FH} steps={FSTEPS} cfg={CFG} seed={seed} ...")
print(f"prompt: {PROMPT[:200]}")

pid, entry = queue_and_wait(wf)
img, server_fn = fetch_image(entry["outputs"])
print(f"done prompt_id={pid} server_file={server_fn} size={img.size}")

out_path = f"/kaggle/working/qwen21_uncensored_{pid[:8]}_s{seed}.png"
img.save(out_path)
print(f"saved → {out_path}")

plt.figure(figsize=(8, 8))
plt.imshow(img)
plt.axis("off")
plt.title(f"s={seed} {FW}x{FH} {FSTEPS} steps", fontsize=10)
plt.show()
"""

code7 = """# ===== 6) Reusable generate() — run this cell for more images =====
import json, time, io, urllib.request, urllib.parse, random
from PIL import Image
import matplotlib.pyplot as plt

def generate(prompt, width=768, height=768, steps=20, cfg=1.0,
             sampler="euler", scheduler="simple", seed=None,
             negative_prompt="", prefix="qwen21_uncensored_t4"):
    seed = seed if isinstance(seed, int) else random.randint(0, 2**31 - 1)
    wf = build_workflow(prompt, DIFFUSION_FILE, TEXT_ENCODER_FILE, VAE_FILE,
                        width, height, steps, cfg, sampler, scheduler, seed)
    pid, entry = queue_and_wait(wf)
    img, _ = fetch_image(entry["outputs"])
    path = f"/kaggle/working/{prefix}_{pid[:8]}_s{seed}.png"
    img.save(path)
    print(f"saved → {path}  ({img.size[0]}x{img.size[1]}, seed={seed})")
    plt.figure(figsize=(8, 8)); plt.imshow(img); plt.axis("off")
    plt.title(prompt[:90], fontsize=10); plt.show()
    return path, img

# Example second image (edit + re-run):
# generate("ultra-detailed macro photo of a dewdrop on a leaf, morning light, bokeh", seed=7)
print("generate() ready. Example:")
print('  generate("cinematic wide shot, misty mountain temple at dawn", seed=42)')
"""

code_tunnel = """# ===== 7) Public tunnels: mini ChatGPT-style UI + ComfyUI direct =====
import os, re, shutil, subprocess, time

if not USE_TUNNEL:
    print("USE_TUNNEL=False, tunnel skipped.")
else:
    if shutil.which("cloudflared") is None:
        print("installing cloudflared (no signup needed)...")
        r = subprocess.run(
            "wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/"
            "cloudflared-linux-amd64 -O /tmp/cloudflared && chmod +x /tmp/cloudflared && "
            "mv /tmp/cloudflared /usr/local/bin/cloudflared",
            shell=True)
        assert r.returncode == 0, "cloudflared download failed"
    else:
        print("cloudflared already installed.")

    TUNNEL_LOG = "/kaggle/working/cloudflared.log"
    subprocess.run("pkill -f 'cloudflared tunnel' 2>/dev/null || true", shell=True)
    time.sleep(1)
    tlog = open(TUNNEL_LOG, "w")
    tproc = subprocess.Popen(
        ["cloudflared", "tunnel", "--url", f"http://127.0.0.1:{MINI_PORT}", "--no-autoupdate"],
        stdout=tlog, stderr=subprocess.STDOUT)
    print(f"cloudflared PID={tproc.pid} (mini UI :{MINI_PORT}), waiting for public URL...")

    public_url = None
    for i in range(60):
        time.sleep(2)
        try:
            txt = open(TUNNEL_LOG).read()
        except Exception:
            continue
        m = re.search(r"https://[a-zA-Z0-9-]+\\.trycloudflare\\.com", txt)
        if m:
            public_url = m.group(0)
            break
    if public_url:
        print("\\n==================================================")
        print(f"  MINI UI (ChatGPT-style, open this):  {public_url}")
        print("==================================================")
        # Report the URL out (Kaggle hides run logs until completion):
        # post to a private ntfy.sh topic so the operator can pick it up live.
        try:
            import urllib.request as _urlreq
            _urlreq.urlopen(_urlreq.Request(
                "https://ntfy.sh/YOUR_PRIVATE_NTFY_TOPIC",
                data=f"miniui={public_url}".encode(), method="POST"), timeout=20)
            print("URL reported to status channel.")
        except Exception as e:
            print(f"status-channel post failed (tunnel still works): {e}")
    # --- Backup tunnel: pinggy (free, no signup; often faster than trycloudflare) ---
    PINGGY_LOG = "/kaggle/working/pinggy.log"
    subprocess.run("pkill -f 'a.pinggy.io' 2>/dev/null || true", shell=True)
    time.sleep(1)
    plog = open(PINGGY_LOG, "w")
    pproc = subprocess.Popen(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30",
         "-N", "-T", "-p", "443", "-R0:127.0.0.1:" + str(COMFY_PORT), "a.pinggy.io"],
        stdout=plog, stderr=subprocess.STDOUT)
    print(f"pinggy ssh PID={pproc.pid}, waiting for public URL...")
    pinggy_url = None
    for i in range(45):
        time.sleep(2)
        try:
            txt = open(PINGGY_LOG).read()
        except Exception:
            continue
        m = re.search(r"https?://[a-zA-Z0-9.-]+\\.pinggy\\.link", txt)
        if m:
            pinggy_url = m.group(0)
            break
    if pinggy_url:
        print("\\n==================================================")
        print(f"  ComfyUI direct (advanced/API):  {pinggy_url}")
        print("==================================================")
        try:
            import urllib.request as _urlreq
            _urlreq.urlopen(_urlreq.Request(
                "https://ntfy.sh/YOUR_PRIVATE_NTFY_TOPIC",
                data=f"comfy={pinggy_url}".encode(), method="POST"), timeout=20)
            print("pinggy URL reported to status channel.")
        except Exception as e:
            print(f"status-channel post failed (tunnel still works): {e}")
    else:
        print("pinggy URL not found — check /kaggle/working/pinggy.log; use cloudflared URL.")
        try:
            print(open(PINGGY_LOG).read()[-1500:])
        except Exception as e:
            print(e)
"""

code_keepalive = """# ===== 8) Keep-alive (INTERACTIVE tunnel sessions only) =====
# Leave KEEP_ALIVE_MINUTES = 0 for normal/batch runs so the run finishes.
# For website-mode use: open this notebook in the Kaggle editor, set
# KEEP_ALIVE_MINUTES = 300 (or more), Run All — the session stays up and the
# tunnel URL keeps working until the minutes elapse or Kaggle ends the session.
import time

if KEEP_ALIVE_MINUTES <= 0:
    print("keep-alive OFF. (Set KEEP_ALIVE_MINUTES > 0 only for interactive tunnel use.)")
else:
    print(f"keep-alive ON for {KEEP_ALIVE_MINUTES} min. Tunnel stays up. Interrupt to stop early.")
    t0 = time.time()
    while (time.time() - t0) < KEEP_ALIVE_MINUTES * 60:
        el = int(time.time() - t0)
        print(f"alive {el//60}m — ComfyUI :{COMFY_PORT}, outputs in /kaggle/working/", flush=True)
        time.sleep(60)
    print("keep-alive finished.")
"""

code_miniui = """# ===== 6b) Mini ChatGPT-style UI (port 8189, tiny page, works on slow tunnels) =====
import json as _json, threading as _th, time as _time, uuid as _uuid
import urllib.request as _req, urllib.parse as _parse
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

MINI_PORT = 8189
JOBS = {}  # job_id -> {status, png, error, prompt}

HTML_PAGE = '''<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Qwen T4 Image</title>
<style>body{font-family:sans-serif;max-width:640px;margin:24px auto;padding:0 12px;background:#111;color:#eee}
textarea{width:100%;box-sizing:border-box;background:#222;color:#eee;border:1px solid #444;border-radius:8px;padding:10px}
button{background:#4a7dff;color:#fff;border:0;border-radius:8px;padding:10px 22px;font-size:16px;margin-top:8px}
select,input{background:#222;color:#eee;border:1px solid #444;border-radius:6px;padding:6px}
img{max-width:100%;border-radius:8px;margin-top:12px}#st{margin-top:10px;color:#9db4ff}</style></head><body>
<h2>Qwen-Image-2.1 (T4)</h2>
<textarea id="p" rows="3" placeholder="Describe the image... e.g. a fox in a snowy forest at night"></textarea><br>
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
</script></body></html>'''

def _mini_run_job(jid, prompt, size, steps):
    JOBS[jid]["t0"] = _time.time()
    try:
        seed = __import__("random").randint(0, 2**31 - 1)
        wf = build_workflow(prompt, DIFFUSION_FILE, TEXT_ENCODER_FILE, VAE_FILE,
                            size, size, steps, 1.0, "euler", "simple", seed)
        pid, entry = queue_and_wait(wf, timeout_s=3600, poll_s=5)
        img, _ = fetch_image(entry["outputs"])
        import io as _io
        buf = _io.BytesIO()
        img.save(buf, format="PNG")
        JOBS[jid].update(status="done", png=buf.getvalue())
    except Exception as e:
        JOBS[jid].update(status="error", error=str(e)[:500])

class _H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body, ctype="text/html"):
        if isinstance(body, str): body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def do_GET(self):
        if self.path == "/":
            return self._send(200, HTML_PAGE)
        q = _parse.urlparse(self.path)
        qs = _parse.parse_qs(q.query)
        if q.path == "/api/status":
            j = JOBS.get(qs.get("id", [""])[0])
            if not j: return self._send(404, "no job", "text/plain")
            el = int(_time.time() - j.get("t0", _time.time()))
            return self._send(200, _json.dumps({"status": j["status"], "elapsed": el,
                "error": j.get("error", "")}), "application/json")
        if q.path == "/api/image":
            j = JOBS.get(qs.get("id", [""])[0])
            if not j or j["status"] != "done": return self._send(404, "not ready", "text/plain")
            return self._send(200, j["png"], "image/png")
        return self._send(404, "?", "text/plain")
    def do_POST(self):
        if self.path != "/api/generate":
            return self._send(404, "?", "text/plain")
        try:
            data = _json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
        except Exception:
            return self._send(400, "bad json", "text/plain")
        prompt = str(data.get("prompt", "")).strip()[:2000]
        if not prompt: return self._send(400, "empty prompt", "text/plain")
        size = int(data.get("size", 768) or 768)
        size = min(max(size, 256), 1536) // 32 * 32
        steps = min(max(int(data.get("steps", 20) or 20), 5), 40)
        jid = _uuid.uuid4().hex[:12]
        JOBS[jid] = {"status": "running", "png": None, "error": "", "t0": _time.time()}
        _th.Thread(target=_mini_run_job, args=(jid, prompt, size, steps), daemon=True).start()
        return self._send(200, _json.dumps({"job_id": jid}), "application/json")

_mini_server = ThreadingHTTPServer(("127.0.0.1", MINI_PORT), _H)
_th.Thread(target=_mini_server.serve_forever, daemon=True).start()
print(f"Mini UI serving on 127.0.0.1:{MINI_PORT} (page is ~4KB, loads even on slow tunnels)")
"""

cells = [
    nbf.v4.new_markdown_cell(md0),
    nbf.v4.new_code_cell(code1),
    nbf.v4.new_code_cell(code2),
    nbf.v4.new_code_cell(code3),
    nbf.v4.new_code_cell(code4),
    nbf.v4.new_code_cell(code5),
    nbf.v4.new_code_cell(code6),
    nbf.v4.new_code_cell(code_miniui),
    nbf.v4.new_code_cell(code_tunnel),
    nbf.v4.new_code_cell(code7),
    nbf.v4.new_code_cell(code_keepalive),
]
for i, c in enumerate(cells):
    if c.cell_type == "code":
        c.metadata.update({"execution": {}})
        c.outputs = []
        c.execution_count = None

nb.cells = cells
with open(NOTEBOOK_PATH, "w") as f:
    nbf.write(nb, f)
print(f"wrote {NOTEBOOK_PATH} with {len(cells)} cells")
