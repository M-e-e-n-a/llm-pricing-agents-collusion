"""Modal app for the hackathon runs: the same pilot with the vLLM backend in bf16 on one L40S.

NOT YET RUN (the Modal credits are reserved for 23-25 Oct). Expected first-run issues to check:
the vLLM image build (vllm==0.32.0 pins torch 2.13 and transformers <5.20), and whether
Qwen3.5-9B loads as Qwen3_5ForConditionalGeneration with images disabled.

Usage, from the repo root:
    modal run pilot/modal_app.py --out pilot_v1 --rounds 100
    modal volume get collusion-runs pilot_v1 runs/
"""

import modal

app = modal.App("collusion-pilot")

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .uv_pip_install("vllm==0.32.0", "scipy", "pydantic")
    .env({"HF_HOME": "/cache/hf", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_dir("pilot", "/root/project/pilot")
    .add_local_dir("third_party/sauer", "/root/project/third_party/sauer")
)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("collusion-runs", create_if_missing=True)


@app.function(image=image, gpu="L40S", timeout=3 * 60 * 60, volumes={"/cache": hf_cache, "/runs": runs})
def pilot(args: list[str]):
    import os
    import sys

    os.chdir("/root/project")
    sys.path.insert(0, "/root/project")
    from pilot.run import main

    try:
        main(["--backend", "vllm", *args])
    finally:
        runs.commit()


@app.local_entrypoint()
def run(out: str, rounds: int = 100, conditions: str = "nochat,chat_en,chat_te,instructed_en",
        seeds: str = "3,3,1,1", memory: int = 100):
    pilot.remote(["--conditions", conditions, "--seeds", seeds, "--rounds", str(rounds),
                  "--memory", str(memory), "--out", f"/runs/{out}"])
