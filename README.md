# Qwen-Image-2.1 (Uncensored GGUF) on Kaggle T4 — with a ChatGPT-style front end

![MIT](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Kaggle T4](https://img.shields.io/badge/Kaggle-T4%20GPU-orange)
![ComfyUI](https://img.shields.io/badge/ComfyUI-GGUF-purple)

We run [`abenzerps/Qwen-Image-2.1-Uncensored-GGUF`](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF)
on a free Kaggle **T4** via **ComfyUI + ComfyUI-GGUF**, and expose it through a tunnel as a dead-simple
website: **type a prompt → get an image**. No ComfyUI graph-building required for daily use.

![Demo — generated on a free Kaggle T4, 768×768, 20 steps](assets/demo.png)
*Demo: "majestic lion portrait at golden hour" — generated remotely through the tunnel.*

Everything here was built iteratively and verified end-to-end with the Kaggle CLI
(push → run → logs): full run, first image, tunnel serving, and remote generation.

## How it works

```
browser / client/generate.py
        │  HTTPS (cloudflared or pinggy tunnel, ~8h sessions)
        ▼
server/mini_ui.py  (:8189, ~4KB page, stdlib only)
        │  ComfyUI API on localhost
        ▼
ComfyUI + leejet/ComfyUI-GGUF  (:8188, --lowvram)
  UnetLoaderGGUF (Q4_K_M, 4.6GB, stays in VRAM)
  + CLIPLoader qwen_image (int8 encoder, 9.35GB, CPU offload)
  + VAE (676MB) → KSampler (euler/simple, cfg 1.0) → PNG
```

Why this stack: the GGUF quant is what fits a 16GB T4; the int8 text encoder lives in system RAM
(encoding runs once per prompt, so speed is unaffected); `cfg=1.0` keeps sampling to one pass per step.
Q8_0 is deliberately avoided — upstream warns of a `[136] vs [128]` shape mismatch on some setups.

## Measured numbers (Kaggle 2×T4, verified from run logs)

| Task | Time |
|---|---|
| Setup (ComfyUI clone, deps, ~14.6GB weights) | ~3–4 min |
| ComfyUI boot | ~40–50 s |
| 512² / 10 steps (test) | ~80 s |
| 768² / 20 steps (fast default) | ~4 min |
| 1024² / 25 steps (best) | ~8–9 min |

## Quickstart — Kaggle (recommended)

1. Kaggle → create a notebook → **Accelerator: GPU T4 x2**, **Internet: ON**.
2. Upload `kaggle/qwen-image-2-1-uncensored-gguf-t4.ipynb` (or copy its cells in).
3. Edit cell 0 if you want (`FAST_MODE`, prompt, sizes). **Run All.**
4. The tunnel cell prints a public `miniui` URL (~6 min in). Open it anywhere: text box → Generate.
5. For long sessions set `KEEP_ALIVE_MINUTES` (last cell); otherwise the run ends after one image.

Push/update from the CLI (see `kaggle/`):

```bash
kaggle kernels push -p kaggle -t 32400   # 9h window for tunnel use
kaggle kernels status <user>/<slug>
kaggle kernels logs   <user>/<slug> | tail -5
```

Set `id` in `kaggle/kernel-metadata.json` to `<your-username>/qwen-image-2-1-uncensored-gguf-t4`
(the slug must match what the server derives from the title — see Troubleshooting).
The notebook reports live tunnel URLs to an [ntfy.sh](https://ntfy.sh) topic so you can grab them
via CLI while logs are hidden; replace `YOUR_PRIVATE_NTFY_TOPIC` in the builders/notebook with
your own random topic (`python -c "import secrets; print('t-'+secrets.token_hex(12))"`).

## Quickstart — remote client (no Kaggle tab needed)

While a tunnel run is alive:

```bash
python client/generate.py "a fox in a snowy forest at night" \
  --url https://<your-tunnel> --size 768 --steps 20 --out fox.png
```

## Quickstart — ComfyUI UI (advanced)

Load `comfy/qwen21_gguf_t4_workflow.json` via the ComfyUI menu → Load.
It is pre-wired (GGUF loader → `qwen_image` CLIP → VAE → sampler) and was validated
socket-by-socket against a live server's `/object_info`.

## Repo map

```
kaggle/   notebook + kernel-metadata.json (Kaggle push source of truth)
server/   mini_ui.py — stdlib-only prompt→image front end for ComfyUI
client/   generate.py — remote CLI client for the tunnel
comfy/    one-click UI workflow JSON for the tunnel's ComfyUI
tools/    build_notebook.py, build_workflow.py — generators for the above
```

Regenerate artifacts after editing the builders:

```bash
python tools/build_notebook.py   # → kaggle/*.ipynb
python tools/build_workflow.py   # → comfy/*.json
```

## Troubleshooting (all hit during the build)

- `ModuleNotFoundError: No module named 'trampoline'` on server boot → install full
  ComfyUI `requirements.txt` (not `--no-deps`) + `trampoline torchsde`. Fixed in the notebook.
- `409 Conflict` on `kernels push` → the local metadata `id` slug didn't match the server's
  slug (`2-1` vs `21`). Keep them identical; pull once to see the canonical slug.
- `kernels.get permission denied` on a private kernel → usually a stale/revoked API token;
  generate a fresh one at kaggle.com/settings/api.
- Tunnel page loads forever → trycloudflare throughput can collapse (~80KB/s measured).
  The mini UI page is ~4KB so it still loads; use it instead of the heavy ComfyUI frontend,
  or use the pinggy URL the notebook also prints.

## Safety & rules

The GGUF release ships **no safety checker** — output is purely a function of your prompt.
Kaggle's terms still apply to what you compute and store, and the law still applies to what you
generate (hard lines: minors, real people without consent). Tunnel URLs have no login:
anyone with the link spends your GPU quota — don't share them.

## Credits & license

- Weights: [abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF)
  (Qwen Research License), via [Qwen/Qwen-Image-2.1](https://huggingface.co/Qwen/Qwen-Image-2.1);
  runner [ComfyUI](https://github.com/comfyanonymous/ComfyUI) + [leejet/ComfyUI-GGUF](https://github.com/leejet/ComfyUI-GGUF).
- Our code in this repo (`server/`, `client/`, `tools/`, notebook glue): MIT — see `LICENSE`.
  Model weights are **not** included and stay under their own license.
