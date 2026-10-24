"""DeepSeek-V4-Flash-0731 behind an OpenAI-compatible SGLang server on one Modal B300, for `--backend vllm-server`.

Adapted from Modal's DeepSeek-V4-Flash example (modal.com/docs/examples/deepseek_v4_flash: SGLang nightly image,
same model and revision, one B300 holds the 155 GiB of weights plus KV cache). Changes: no speculative decoding (so
no DSPARK drafter and no drafter patch; the sprint needs throughput across many sessions, not single-stream
latency), thinking on by default (we run reasoning on, like gpt-oss at medium effort), a 16k context instead of
268k, one container at most, and an API key so the public URL only answers our harness. The key is the same Modal
secret `vllm-api-key` the gpt-oss server uses (`.env`: VLLM_API_KEY).

Usage, from the repo root:
    modal deploy pilot/modal_dsv4.py               # first deploy builds the image and downloads the weights (CPU)
    modal app stop dsv4-server                     # stop billing when a run is done
"""

import json
import shlex
import subprocess
import time
import urllib.error
import urllib.request

import modal

MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
REVISION = "9e165c30e2704aec5d9d593cce3eebd58bbef1cb"  # the revision Modal's example pins
MINUTES = 60
PORT = 8000
MAX_RUNNING = 128  # concurrent requests; a fork step sends one per branch (up to ~120)
HF_CACHE_DIR = "/root/.cache/huggingface"
DG_CACHE_DIR = "/cache/deep_gemm"
FLASHINFER_CACHE_DIR = "/root/.cache/sglang/flashinfer"

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
dg_cache = modal.Volume.from_name("sglang-deepgemm-cache", create_if_missing=True)
flashinfer_cache = modal.Volume.from_name("flashinfer-autotune-cache", create_if_missing=True)


def download_model(repo_id, revision):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, revision=revision, max_workers=16)


image = (
    modal.Image.from_registry("lmsysorg/sglang:nightly-dev-cu13-20260729-16a52bff")
    .entrypoint([])
    .run_commands("rm -rf /root/.cache/huggingface")
    .env({"HF_HUB_CACHE": HF_CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
    .run_function(download_model, volumes={HF_CACHE_DIR: hf_cache}, args=(MODEL, REVISION), timeout=4 * 60 * MINUTES,
                  cpu=8)
    .env({
        "SGLANG_DG_CACHE_DIR": DG_CACHE_DIR,
        "SGLANG_JIT_DEEPGEMM_FAST_WARMUP": "1",
        "TILELANG_CACHE_DIR": f"{DG_CACHE_DIR}/tilelang",
        "NCCL_CUMEM_ENABLE": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SGLANG_TIMEOUT_KEEP_ALIVE": f"{5 * MINUTES}",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
    })
)

app = modal.App("dsv4-server")


def server_command(api_key: str) -> list[str]:
    """Flags from the example (SGLang's DeepSeek-V4 cookbook, B300, FP4) minus the speculative-decoding ones."""
    return ["sglang", "serve", "--model-path", MODEL, "--served-model-name", MODEL, "--revision", REVISION,
            "--host", "0.0.0.0", "--port", str(PORT), "--api-key", api_key, "--tp", "1",
            "--chunked-prefill-size", "4096", "--context-length", "16384",
            "--max-running-requests", str(MAX_RUNNING), "--cuda-graph-max-bs-decode", str(MAX_RUNNING),
            "--decode-log-interval", "200", "--default-chat-template-kwargs", json.dumps({"thinking": True}),
            "--disable-flashinfer-autotune", "--dist-timeout", f"{60 * MINUTES}", "--mem-fraction-static", "0.90",
            "--moe-a2a-backend", "none", "--moe-runner-backend", "flashinfer_mxfp4", "--reasoning-parser",
            "deepseek-v4", "--swa-full-tokens-ratio", "0.1", "--trust-remote-code", "--skip-server-warmup"]


def wait_ready(proc: subprocess.Popen) -> None:
    while True:
        if proc.poll() is not None:
            raise RuntimeError(f"SGLang exited with code {proc.returncode} before becoming healthy")
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    print("server is ready")
                    return
        except (urllib.error.URLError, OSError, TimeoutError):
            pass
        time.sleep(5)


@app.server(image=image, gpu="B300:1", cpu=8, memory=96 * 1024, port=PORT, startup_timeout=60 * MINUTES,
            volumes={HF_CACHE_DIR: hf_cache, DG_CACHE_DIR: dg_cache, FLASHINFER_CACHE_DIR: flashinfer_cache},
            secrets=[modal.Secret.from_name("vllm-api-key")], min_containers=0, max_containers=1,
            target_concurrency=MAX_RUNNING, scaledown_window=10 * MINUTES, unauthenticated=True)
class Server:
    @modal.enter()
    def startup(self):
        import os

        key = os.environ["VLLM_API_KEY"]
        cmd = server_command(key)
        print(shlex.join("***" if a == key else a for a in cmd))
        self.proc = subprocess.Popen(cmd, start_new_session=True)
        wait_ready(self.proc)

    @modal.exit()
    def stop(self):
        self.proc.terminate()
        self.proc.wait()
