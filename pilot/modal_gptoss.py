"""gpt-oss-120b behind an OpenAI-compatible vLLM server on one Modal H200, for `--backend vllm-server`.

Adapted from Modal's gpt-oss example (modal.com/docs/examples/gpt_oss_inference, vLLM 0.18.1). Changes: the 120b
model pinned to a revision, an H200 instead of a B200 (the Blackwell-only MoE flag dropped; an 80 GB H100 ran out
of memory while repacking the MXFP4 weights), no speculative decoding (the sprint needs throughput across many
sessions, not single-stream latency), and an API key so the public URL only
answers our harness. The key lives in the Modal secret `vllm-api-key` and in `.env` as VLLM_API_KEY.

Usage, from the repo root:
    modal run pilot/modal_gptoss.py::download      # fetch the weights into the volume on a CPU container
    modal deploy pilot/modal_gptoss.py             # prints the server URL; base URL = <that URL>/v1
    modal app stop gptoss-server                   # stop billing when a run is done
"""

import json

import modal

MODEL = "openai/gpt-oss-120b"
REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MINUTES = 60
PORT = 8000
MAX_INPUTS = 64  # concurrent requests per replica; prompts are ~2.3k tokens in, up to 3k out

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.18.1", "huggingface_hub[hf_transfer]==0.36.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
)
hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
volumes = {"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache}

app = modal.App("gptoss-server")


@app.function(image=image, volumes=volumes, timeout=60 * MINUTES)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL, revision=REVISION, ignore_patterns=["original/*", "metal/*"])
    hf_cache.commit()


@app.function(image=image, gpu="H200", volumes=volumes, secrets=[modal.Secret.from_name("vllm-api-key")],
              scaledown_window=10 * MINUTES, timeout=30 * MINUTES)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(port=PORT, startup_timeout=30 * MINUTES)
def serve():
    import os
    import subprocess

    cmd = ["vllm", "serve", MODEL, "--revision", REVISION, "--served-model-name", MODEL,
           "--host", "0.0.0.0", "--port", str(PORT), "--api-key", os.environ["VLLM_API_KEY"],
           "--max-model-len", "16384", "--max-num-seqs", str(MAX_INPUTS),
           "--compilation-config", json.dumps({"pass_config": {"eliminate_noops": True}})]
    print(*("***" if a == os.environ["VLLM_API_KEY"] else a for a in cmd))
    subprocess.Popen(cmd)
