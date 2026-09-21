"""Build one-click ComfyUI UI-format workflow for the live T4 tunnel."""
import json

PROMPT = "cinematic portrait of a warrior queen at sunset, highly detailed, dramatic lighting, 35mm"
W = H = 768
STEPS, CFG, SEED = 20, 1.0, 123456

def inp(name, typ, link=None, widget=None):
    d = {"name": name, "type": typ, "link": link}
    if widget:
        d["widget"] = {"name": widget}
    return d

def out(name, typ, links, slot):
    return {"name": name, "type": typ, "links": links, "slot_index": slot}

def node(i, typ, x, y, w, h, inputs, outputs, widgets):
    return {"id": i, "type": typ, "pos": [x, y], "size": [w, h],
            "flags": {}, "order": i - 1, "mode": 0,
            "inputs": inputs, "outputs": outputs,
            "properties": {"Node name for S&R": typ},
            "widgets_values": widgets}

nodes = [
    node(1, "UnetLoaderGGUF", 40, 80, 340, 110,
         [inp("unet_name", "COMBO", widget="unet_name")],
         [out("MODEL", "MODEL", [1], 0)],
         ["qwen-image-2.1-Q4_K_M.gguf"]),
    node(2, "CLIPLoader", 40, 240, 340, 150,
         [inp("clip_name", "COMBO", widget="clip_name"),
          inp("type", "COMBO", widget="type"),
          inp("device", "COMBO", widget="device")],
         [out("CLIP", "CLIP", [2], 0)],
         ["qwen3vl_8b_int8_convrot.safetensors", "qwen_image", "default"]),
    node(3, "VAELoader", 40, 440, 340, 100,
         [inp("vae_name", "COMBO", widget="vae_name")],
         [out("VAE", "VAE", [3], 0)],
         ["qwen_image_2.1_vae_bf16.safetensors"]),
    node(4, "TextEncodeQwenImage21", 450, 80, 420, 320,
         [inp("clip", "CLIP", link=2),
          inp("prompt", "STRING", widget="prompt"),
          inp("negative_prompt", "STRING", widget="negative_prompt"),
          inp("resolution", "INT", widget="resolution")],
         [out("positive", "CONDITIONING", [4], 0),
          out("negative", "CONDITIONING", [5], 1),
          out("latent", "LATENT", None, 2)],
         [PROMPT, "", 768]),
    node(5, "EmptyLatentImage", 450, 450, 330, 120,
         [inp("width", "INT", widget="width"),
          inp("height", "INT", widget="height"),
          inp("batch_size", "INT", widget="batch_size")],
         [out("LATENT", "LATENT", [6], 0)],
         [W, H, 1]),
    node(6, "KSampler", 930, 120, 330, 270,
         [inp("model", "MODEL", link=1),
          inp("positive", "CONDITIONING", link=4),
          inp("negative", "CONDITIONING", link=5),
          inp("latent_image", "LATENT", link=6),
          inp("seed", "INT", widget="seed"),
          inp("steps", "INT", widget="steps"),
          inp("cfg", "FLOAT", widget="cfg"),
          inp("sampler_name", "COMBO", widget="sampler_name"),
          inp("scheduler", "COMBO", widget="scheduler"),
          inp("denoise", "FLOAT", widget="denoise")],
         [out("LATENT", "LATENT", [7], 0)],
         [SEED, STEPS, CFG, "euler", "simple", 1.0]),
    node(7, "VAEDecode", 1330, 150, 240, 80,
         [inp("samples", "LATENT", link=7),
          inp("vae", "VAE", link=3)],
         [out("IMAGE", "IMAGE", [8], 0)],
         []),
    node(8, "SaveImage", 1330, 280, 330, 200,
         [inp("images", "IMAGE", link=8)],
         [],
         ["qwen21_uncensored_t4"]),
]

links = [
    [1, 1, 0, 6, 0, "MODEL"],
    [2, 2, 0, 4, 0, "CLIP"],
    [3, 3, 0, 7, 1, "VAE"],
    [4, 4, 0, 6, 1, "CONDITIONING"],
    [5, 4, 1, 6, 2, "CONDITIONING"],
    [6, 5, 0, 6, 3, "LATENT"],
    [7, 6, 0, 7, 0, "LATENT"],
    [8, 7, 0, 8, 0, "IMAGE"],
]

wf = {"version": 0.4, "nodes": nodes, "links": links,
      "groups": [], "config": {}, "extra": {}}

# integrity check: every link endpoint exists with matching slot/type
by_id = {n["id"]: n for n in nodes}
for lid, sn, ss, dn, ds, t in links:
    s = by_id[sn]["outputs"][ss]
    assert lid in (s["links"] or []), f"link {lid} missing at src"
    assert s["type"] == t, f"src type mismatch link {lid}"
    d = by_id[dn]["inputs"][ds]
    assert d["link"] == lid and d["type"] == t, f"dst mismatch link {lid}"
print("link integrity OK")

OUT = "/Users/owner/Desktop/alphaxiz/uncensored/qwen21_gguf_t4_workflow.json"
with open(OUT, "w") as f:
    json.dump(wf, f, indent=1)
print(f"wrote {OUT} ({len(json.dumps(wf))} bytes)")
