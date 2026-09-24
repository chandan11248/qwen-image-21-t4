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

**Model files used (T4-fit; turbo-verified 2026-09-21 — warm runs do 768/20 in ~5s):**
- Diffusion: `qwen-image-2.1-UC-int8_convrot.safetensors` — ~7.3 GB official-style int8 (standard loader, fastest cold start; GGUF `UC-Q4_K_M` fallback available)
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

# Diffusion GGUF choice (upstream renamed files with UC- prefix; Q4_K_M recommended):
#   "qwen-image-2.1-UC-Q4_K_M.gguf" (4.60GB) | "qwen-image-2.1-UC-Q4_0.gguf" (4.05GB)
#   "qwen-image-2.1-UC-Q5_K_M.gguf" (5.22GB) | "qwen-image-2.1-UC-Q6_K.gguf" (5.88GB)
DIFFUSION_FILE = "qwen-image-2.1-UC-Q4_K_M.gguf"

# Serving diffusion: "int8" (official UC-int8, faster cold start, standard loader)
# or "gguf" (smaller download, more VRAM headroom). Benchmark verdict 2026-09-21:
# warm sampling tied at ~5s per 768/20 image; cold start int8 162s vs gguf 241s; quality tied.
SERVE_DIFFUSION = "int8"

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
# Reliable tunnel: free ngrok authtoken (https://dashboard.ngrok.com/get-started/your-authtoken).
# Paste it here (or leave empty to skip ngrok and use cloudflared+pinggy only).
NGROK_TOKEN = ""
# Keep-alive loop minutes (holds the run + tunnel open; 0 = off so batch runs finish)
KEEP_ALIVE_MINUTES = 420  # ~7h persistent tunnel (fits remaining quota)

# ---- TURBO benchmark (timed A/B) ----
# Official-style int8 diffusion now ships IN the uncensored repo (root level).
INT8_DIFF_REPO = REPO_ID
INT8_DIFF_FILE = "qwen-image-2.1-UC-int8_convrot.safetensors"  # ~7.3GB, repo root
FETCH_INT8 = False   # serving pulls int8 via SERVE_DIFFUSION; bench-int8 run flips True
RUN_BENCH = False    # True → timed benchmark run, then finish
RUN_EVAL = False     # True → full speed+quality evaluation across sizes
MAKE_ART = False     # True → generate 4 art pieces + GIF + upload (eval runs)
RUN_SAMPLER = False  # True → sampler/step shootout with PNG uploads for visual judging
BENCH_MODE = "gguf"  # backend under test: "gguf" or "int8" (one diffusion per run: 20GB cap)
EVAL_SEED = 777
BENCH_PROMPT = "cinematic portrait of a warrior queen at sunset, highly detailed, dramatic lighting, 35mm"
BENCH_SEED = 123456

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
print("--- df -h ---")
print(subprocess.run(["df", "-h", "/kaggle/working", "/tmp"],
      capture_output=True, text=True, timeout=20).stdout)

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

# Kaggle image already ships the torch stack + most deps — NEVER reinstall torch here
# (ComfyUI's requirements.txt would drag multi-GB torch/nvidia wheels and fill the 20GB disk).
# Install ONLY what is actually missing, with no pip cache.
import importlib.util as _ilu
NEED = {"gguf": "gguf", "trampoline": "trampoline", "torchsde": "torchsde",
        "websocket-client": "websocket", "requests": "requests", "einops": "einops",
        "safetensors": "safetensors", "aiohttp": "aiohttp", "kornia": "kornia",
        "spandrel": "spandrel", "soundfile": "soundfile", "av": "av",
        "pyyaml": "yaml", "tqdm": "tqdm", "pillow": "PIL"}
_missing = [pkg for pkg, mod in NEED.items() if _ilu.find_spec(mod) is None]
print("torch stack present:",
      all(_ilu.find_spec(m) is not None for m in ["torch", "torchvision"]))
for _m in ["torchvision", "torchaudio"]:
    if _ilu.find_spec(_m) is None:
        # --no-deps: take the lib WITHOUT dragging a second torch copy
        run(f"{sys.executable} -m pip install -q --no-cache-dir --no-deps {_m} || true")
if _missing:
    run(f"{sys.executable} -m pip install -q --no-cache-dir {' '.join(_missing)}")
else:
    print("all small deps present, pip install skipped.")

if not os.path.isdir(COMFY_DIR):
    run(f"git clone --depth 1 https://github.com/comfyanonymous/ComfyUI {COMFY_DIR}")
else:
    print(f"ComfyUI exists: {COMFY_DIR}, pulling latest (for Qwen-Image-2.1 nodes)...")
    run(f"git -C {COMFY_DIR} pull --ff-only || true")

# ComfyUI python deps (full install: Kaggle's torch 2.10+cu128 already satisfies torch pins,
# and full install pulls transitive deps like `trampoline` required by torchsde)
run(f"{sys.executable} -m pip install -q --no-cache-dir -r {COMFY_DIR}/requirements.txt || true")
# Belt-and-braces for the torchsde->trampoline import chain (missing this broke server boot on v1):
run(f"{sys.executable} -m pip install -q --no-cache-dir trampoline torchsde || true")

GGUF_NODE_DIR = os.path.join(COMFY_DIR, "custom_nodes", "ComfyUI-GGUF")
if not os.path.isdir(GGUF_NODE_DIR):
    run(f"git clone --depth 1 https://github.com/leejet/ComfyUI-GGUF {GGUF_NODE_DIR}")
else:
    run(f"git -C {GGUF_NODE_DIR} pull --ff-only || true")
run(f"{sys.executable} -m pip install -q -r {GGUF_NODE_DIR}/requirements.txt || true")

print("\\nComfyUI + ComfyUI-GGUF ready.")
print(f"ComfyUI: {COMFY_DIR}")
"""

code4 = """# ===== 3) Download weights (all streamed via wget: no 2x HF/XET cache on 20GB disk) =====
import os
import urllib.parse as _up

def _wget(repo, remote, dest, min_gb):
    url = f"https://huggingface.co/{repo}/resolve/main/{_up.quote(remote)}"
    print(f"Downloading {remote} ...", flush=True)
    r = subprocess.run(f"wget -q -c -O {dest} '{url}'", shell=True, capture_output=False)
    gb = os.path.getsize(dest) / 1024**3 if os.path.exists(dest) else 0
    assert r.returncode == 0 and gb > min_gb, f"download failed: {remote}"
    print(f"{gb:5.2f} GB  {dest}", flush=True)
    return dest

diff_dir = os.path.join(COMFY_DIR, "models", "diffusion_models")
te_dir   = os.path.join(COMFY_DIR, "models", "text_encoders")
vae_dir  = os.path.join(COMFY_DIR, "models", "vae")
for d in [diff_dir, te_dir, vae_dir]:
    os.makedirs(d, exist_ok=True)

print(f"SERVING diffusion={SERVE_DIFFUSION} RUN_BENCH={RUN_BENCH} RUN_EVAL={RUN_EVAL} BENCH_MODE={BENCH_MODE}")
p_diff = None
# Eval runs test SERVE_DIFFUSION; only bench runs use BENCH_MODE to add a second file.
# (One diffusion per run — the 20GB disk cap bit us when both flags pulled GGUF+int8.)
if SERVE_DIFFUSION == "gguf" or (RUN_BENCH and BENCH_MODE == "gguf"):
    p_diff = _wget(REPO_ID, DIFFUSION_FILE, os.path.join(diff_dir, DIFFUSION_FILE), 3.5)
else:
    print("Skipping GGUF diffusion (serving/eval int8).")
p_te = _wget(REPO_ID, f"text_encoders/{TEXT_ENCODER_FILE}",
             os.path.join(te_dir, TEXT_ENCODER_FILE), 8.0)
p_vae = _wget(REPO_ID, f"vae/{VAE_FILE}", os.path.join(vae_dir, VAE_FILE), 0.5)

if FETCH_INT8 or SERVE_DIFFUSION == "int8":
    _wget(INT8_DIFF_REPO, INT8_DIFF_FILE, os.path.join(diff_dir, INT8_DIFF_FILE), 5.0)

print("--- disk after downloads ---")
print(subprocess.run(["df", "-h", "/kaggle/working"],
      capture_output=True, text=True, timeout=20).stdout)

print("\\n--- files ---")
for p in [x for x in [p_diff, p_te, p_vae] if x]:
    sz = os.path.getsize(p)/1024**3
    print(f"{sz:5.2f} GB  {p}")

# Sanity: filenames the API workflow will request must exist
if p_diff is not None:
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

def build_workflow(prompt, diffusion_file, te_file, vae_file, w, h, steps, cfg, sampler, scheduler, seed, loader_node=None):
    # Flat API workflow — avoids version-fragile subgraph templates.
    # Node classes: UnetLoaderGGUF (leejet/ComfyUI-GGUF) or UNETLoader (safetensors),
    #   CLIPLoader(type=qwen_image), VAELoader, TextEncodeQwenImage21,
    #   EmptyLatentImage, KSampler, VAEDecode, SaveImage
    if loader_node is None:
        if SERVE_DIFFUSION == "int8":
            loader_node = {"class_type": "UNETLoader",
                           "inputs": {"unet_name": INT8_DIFF_FILE, "weight_dtype": "default"}}
        else:
            loader_node = {"class_type": "UnetLoaderGGUF",
                           "inputs": {"unet_name": diffusion_file}}
    return {
        "1": loader_node,
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
_loader = None
if RUN_BENCH and BENCH_MODE == "int8":
    _loader = {"class_type": "UNETLoader",
               "inputs": {"unet_name": INT8_DIFF_FILE, "weight_dtype": "default"}}
wf = build_workflow(PROMPT, DIFFUSION_FILE, TEXT_ENCODER_FILE, VAE_FILE,
                    FW, FH, FSTEPS, CFG, SAMPLER, SCHEDULER, seed, _loader)
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
    # --- Primary tunnel: ngrok (needs free authtoken; most reliable). ---
    # Set NGROK_TOKEN in config (cell 0). Get one free at https://dashboard.ngrok.com/get-started/your-authtoken
    if NGROK_TOKEN.strip():
        subprocess.run("pkill -f 'ngrok http' 2>/dev/null || true", shell=True)
        time.sleep(1)
        if shutil.which("ngrok") is None:
            print("installing ngrok...")
            r = subprocess.run("wget -q https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -O /tmp/ngrok.tgz "
                               "&& tar xzf /tmp/ngrok.tgz -C /usr/local/bin/", shell=True)
            assert r.returncode == 0, "ngrok download failed"
        ngrok_log = "/kaggle/working/ngrok.log"
        _nl = open(ngrok_log, "w")
        _np = subprocess.Popen(["ngrok", "http", str(MINI_PORT), "--log", "stdout"],
                               stdout=_nl, stderr=subprocess.STDOUT,
                               env={**os.environ, "NGROK_AUTHTOKEN": NGROK_TOKEN.strip()})
        print(f"ngrok PID={_np.pid}, waiting for public URL...")
        ngrok_url = None
        for i in range(45):
            time.sleep(2)
            try:
                txt = open(ngrok_log).read()
            except Exception:
                continue
            m = re.search(r"https://[a-zA-Z0-9.-]+\\.ngrok(?:-free)?\\.app", txt)
            if m:
                ngrok_url = m.group(0)
                break
        if ngrok_url:
            print("\\n==================================================")
            print(f"  NGROK URL (use this):  {ngrok_url}")
            print("==================================================")
            try:
                import urllib.request as _urlreq
                _urlreq.urlopen(_urlreq.Request(
                    "https://ntfy.sh/YOUR_PRIVATE_NTFY_TOPIC",
                    data=f"ngrok={ngrok_url}".encode(), method="POST"), timeout=20)
            except Exception as e:
                print(f"status-channel post failed (tunnel still works): {e}")
        else:
            print("ngrok URL not found — check /kaggle/working/ngrok.log")
    else:
        print("NGROK_TOKEN empty, ngrok skipped (set it for the reliable tunnel).")
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

code_bench = """# ===== 9) TURBO benchmark: GGUF vs official int8, timed A/B =====
import time as _t
import numpy as _np

def _sharpness(img):
    g = _np.asarray(img.convert("L"), dtype=_np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())

def _bench_wf(loader_node, prompt, size, steps, seed):
    return {
        "1": loader_node,
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": TEXT_ENCODER_FILE, "type": "qwen_image"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": VAE_FILE}},
        "4": {"class_type": "TextEncodeQwenImage21",
              "inputs": {"clip": ["2", 0], "prompt": prompt,
                         "negative_prompt": "", "resolution": size}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": size, "height": size, "batch_size": 1}},
        "6": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["4", 1],
                         "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                         "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode",
              "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": "bench_turbo"}},
    }

def _bench_one(name, loader_node, size=768, steps=20):
    wf = _bench_wf(loader_node, BENCH_PROMPT, size, steps, BENCH_SEED)
    t0 = _t.time()
    pid, entry = queue_and_wait(wf, timeout_s=1800, poll_s=5)
    dt = _t.time() - t0
    img, _ = fetch_image(entry["outputs"])
    path = f"/kaggle/working/bench_{name}_{pid[:8]}.png"
    img.save(path)
    print(f"BENCH[{name}] {dt:.0f}s  sharp={_sharpness(img):.0f}  size={img.size}  -> {path}", flush=True)
    return dt

if not RUN_BENCH:
    print("RUN_BENCH=False, turbo benchmark skipped.")
else:
    if BENCH_MODE == "int8":
        assert os.path.exists(os.path.join(COMFY_DIR, "models", "diffusion_models", INT8_DIFF_FILE)), "int8 file missing"
        loader = {"class_type": "UNETLoader", "inputs": {"unet_name": INT8_DIFF_FILE, "weight_dtype": "default"}}
    else:
        loader = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": DIFFUSION_FILE}}
    t = _bench_one(f"{BENCH_MODE}", loader)
    print(f"BENCH RESULT [{BENCH_MODE} 768/20]: {t:.0f}s", flush=True)
"""

code_eval = """# ===== 10) EVAL: speed + quality across sizes (timed, with stats) =====
import time as _t
import numpy as _np

def _img_stats(img):
    g = _np.asarray(img.convert("L"), dtype=_np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    a = _np.asarray(img, dtype=_np.float32)
    return float(lap.var()), float(a.mean()), float(a.std())

def _eval_one(tag, prompt, size, steps, seed):
    wf = build_workflow(prompt, DIFFUSION_FILE, TEXT_ENCODER_FILE, VAE_FILE,
                        size, size, steps, 1.0, "euler", "simple", seed)
    t0 = _t.time()
    pid, entry = queue_and_wait(wf, timeout_s=1800, poll_s=5)
    dt = _t.time() - t0
    img, _ = fetch_image(entry["outputs"])
    path = f"/kaggle/working/eval_{tag}_{pid[:8]}.png"
    img.save(path)
    sh, br, co = _img_stats(img)
    print(f"EVAL[{tag}] {dt:.0f}s sharp={sh:.0f} bright={br:.0f} contrast={co:.0f} size={img.size} -> {path}", flush=True)
    return path, dt

if not RUN_EVAL:
    print("RUN_EVAL=False, evaluation skipped.")
else:
    _P = "cinematic portrait of a warrior queen at sunset, highly detailed, dramatic lighting, 35mm"
    _eval_one(f"{SERVE_DIFFUSION}-cold-768-20", _P, 768, 20, EVAL_SEED)
    _eval_one(f"{SERVE_DIFFUSION}-repeat-same-seed", _P, 768, 20, EVAL_SEED)
    _eval_one(f"{SERVE_DIFFUSION}-warm-768-20", _P, 768, 20, EVAL_SEED + 1)
    _eval_one(f"{SERVE_DIFFUSION}-warm-1024-25", _P, 1024, 25, EVAL_SEED + 2)
    _eval_one(f"{SERVE_DIFFUSION}-warm-512-10", _P, 512, 10, EVAL_SEED + 3)
    print("EVAL done.", flush=True)
"""

code_art = """# ===== 11) ART: 4 pieces -> GIF -> upload =====
import subprocess as _sp

ART_PROMPTS = [
    ("neon cyberpunk city in rain, flying vehicles, reflections", 11),
    ("astronaut riding a white horse on mars, cinematic dust", 22),
    ("underwater crystal palace with whales, god rays, fantasy", 33),
    ("samurai standing in cherry blossom storm, ukiyo-e style", 44),
]

if not MAKE_ART:
    print("MAKE_ART=False, art skipped.")
else:
    _frames = []
    for _pr, _sd in ART_PROMPTS:
        _wf = build_workflow(_pr, DIFFUSION_FILE, TEXT_ENCODER_FILE, VAE_FILE,
                             768, 768, 20, 1.0, "euler", "simple", _sd)
        _pid, _entry = queue_and_wait(_wf, timeout_s=1800, poll_s=5)
        _img, _ = fetch_image(_entry["outputs"])
        _p = f"/kaggle/working/art_{_sd}.png"
        _img.save(_p)
        _frames.append(_img.resize((512, 512)).convert("RGB"))
        print(f"ART {_sd} done -> {_p}", flush=True)
    _gif = "/kaggle/working/qwen_t4_art.gif"
    _frames[0].save(_gif, save_all=True, append_images=_frames[1:], duration=900, loop=0)
    print(f"GIF saved {os.path.getsize(_gif)/1024:.0f} KB -> {_gif}", flush=True)
    _up1 = _sp.run(f"curl -s --max-time 120 -F 'file=@{_gif}' https://0x0.st",
                   shell=True, capture_output=True, text=True, timeout=150)
    print(f"FILE_URL_0X0={_up1.stdout.strip()}", flush=True)
    _up2 = _sp.run(f"curl -s --max-time 120 -F 'reqtype=fileupload' -F 'time=72h' -F 'fileToUpload=@{_gif}' https://litterbox.catbox.moe/resources/internals/api.php",
                   shell=True, capture_output=True, text=True, timeout=150)
    print(f"FILE_URL_LITTERBOX={_up2.stdout.strip()}", flush=True)
"""

code_sampler = """# ===== 12) SAMPLER shootout: fewer steps via better solvers (timed + upload) =====
import time as _t
import numpy as _np
import subprocess as _sp

_SAMPLER_TESTS = [
    ("euler-20",   "euler",   20, 501),
    ("unipc-12",   "uni_pc",  12, 502),
    ("dpmpp2m-12", "dpmpp_2m", 12, 503),
    ("dpmpp2m-10", "dpmpp_2m", 10, 504),
    ("euler-12",   "euler",   12, 505),
]
_SP_PROMPT = "cinematic portrait of a warrior queen at sunset, highly detailed, dramatic lighting, 35mm"

def _sstats(img):
    g = _np.asarray(img.convert("L"), dtype=_np.float32)
    lap = g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]
    return float(lap.var())

if not RUN_SAMPLER:
    print("RUN_SAMPLER=False, sampler shootout skipped.")
else:
    for _tag, _smp, _st, _sd in _SAMPLER_TESTS:
        try:
            _wf = build_workflow(_SP_PROMPT, DIFFUSION_FILE, TEXT_ENCODER_FILE, VAE_FILE,
                                 768, 768, _st, 1.0, _smp, "simple", _sd)
            _t0 = _t.time()
            _pid, _entry = queue_and_wait(_wf, timeout_s=1800, poll_s=5)
            _dt = _t.time() - _t0
            _img, _ = fetch_image(_entry["outputs"])
            _path = f"/kaggle/working/sampler_{_tag}.png"
            _img.save(_path)
            _up = _sp.run(f"curl -s --max-time 120 -F 'reqtype=fileupload' -F 'time=72h' -F 'fileToUpload=@{_path}' https://litterbox.catbox.moe/resources/internals/api.php",
                          shell=True, capture_output=True, text=True, timeout=150)
            print(f"SAMPLER[{_tag}] {_dt:.0f}s sharp={_sstats(_img):.0f} -> {_path}", flush=True)
            print(f"SAMPLER_URL_{_tag}={_up.stdout.strip()}", flush=True)
        except Exception as _e:
            print(f"SAMPLER[{_tag}] FAILED: {str(_e)[:300]}", flush=True)
    print("SAMPLER shootout done.", flush=True)
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
    nbf.v4.new_code_cell(code_bench),
    nbf.v4.new_code_cell(code_eval),
    nbf.v4.new_code_cell(code_art),
    nbf.v4.new_code_cell(code_sampler),
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
