"""Run pilot.fork on a Modal CPU container, so a run does not depend on the laptop staying awake.

Each run is one Function call that shells out to `python -m pilot.fork ... --out /runs/<name>`. Its output lands in
the Modal Volume `fork-runs` (background commits every few seconds), along with driver.log. A run that dies can be
started again with the same arguments: pilot.fork resumes from the branches already recorded.

    modal run --detach pilot/modal_driver.py --plan main      # spawn every run in PLANS["main"] and wait
    modal volume get fork-runs main_gptoss_plain data/fork_runs/   # copy one run back
"""

import modal

MINUTES = 60
GPTOSS = ["--backend", "vllm-server", "--api-model", "openai/gpt-oss-120b",
          "--base-url", "https://pranav-kasetty--gptoss-server-serve.us-east.modal.run/v1"]
DSV4 = ["--backend", "vllm-server", "--api-model", "deepseek-ai/DeepSeek-V4-Flash-0731",
        "--base-url", "https://pranav-kasetty--dsv4-server-server.us-east.modal.direct/v1", "--max-new-tokens", "8000"]
COMMON = ["--repeat-offer", "--first-seed", "100"]
# OpenRouter, one pinned endpoint each. coreweave/fp4 serves gpt-oss in its native 4-bit format, like the vLLM server.
OR_GPTOSS = ["--backend", "openrouter", "--api-model", "openai/gpt-oss-120b", "--provider", "coreweave/fp4",
             "--reasoning", "medium", "--max-new-tokens", "4000"]
OR_HAIKU = ["--backend", "openrouter", "--api-model", "anthropic/claude-haiku-5.5", "--provider", "anthropic",
            "--reasoning", "medium", "--max-new-tokens", "4000"]
OR_DSV4 = ["--backend", "openrouter", "--api-model", "deepseek/deepseek-v4-flash-0731", "--provider", "deepinfra/fp8",
           "--reasoning", "medium", "--max-new-tokens", "8000"]
DISRUPT_TEXTS = ["--texts", "pilot/fork_texts_disrupt.json"]

# name -> fork arguments. Main runs of prereg-v1 (PREREGISTRATION.md, section 5).
PLANS = {
    "main": {
        "main_gptoss_plain": GPTOSS + ["--agents", "plain", "--sessions", "44", "--workers", "40"],
        "main_gptoss_ia": GPTOSS + ["--agents", "instructed_acceptor", "--sessions", "48", "--workers", "40",
                                    "--versions", "N,R,O,Announce,Hint,Benign"],
        "main_gptoss_known": GPTOSS + ["--agents", "competitive,silent_acceptor,double_crosser,announcer",
                                       "--sessions", "42", "--workers", "48", "--versions", "N,R,O"],
        "main_gptoss_follower": GPTOSS + ["--agents", "follower", "--sessions", "74", "--workers", "24",
                                          "--versions", "N,R,O"],
        "main_gptoss_fa": GPTOSS + ["--agents", "follower_acceptor", "--sessions", "125", "--workers", "40",
                                    "--versions", "N,R,O"],
        "main_dsv4_plain": DSV4 + ["--agents", "plain", "--sessions", "40", "--workers", "80"],
        "main_dsv4_ia": DSV4 + ["--agents", "instructed_acceptor", "--sessions", "14", "--workers", "24",
                                "--versions", "N,R,O"],
    },
    # prereg-v1.1: disruption versions branched from the main runs' saved warm-ups; O and R rerun on the same
    # endpoint as matched controls
    "disrupt": {
        "dis_gptoss_plain": OR_GPTOSS + DISRUPT_TEXTS + [
            "--forks-from", "/runs/main_gptoss_plain", "--agents", "plain", "--sessions", "44", "--workers", "40",
            "--versions", "R,O,R_warn,O_warn,O_reveal", "--max-usd", "3"],
        "dis_gptoss_ia": OR_GPTOSS + DISRUPT_TEXTS + [
            "--forks-from", "/runs/main_gptoss_ia", "--agents", "instructed_acceptor", "--sessions", "48",
            "--workers", "40", "--versions", "O,O_warn,O_reveal", "--max-usd", "2"],
        "dis_gptoss_sa": OR_GPTOSS + DISRUPT_TEXTS + [
            "--forks-from", "/runs/main_gptoss_known", "--agents", "silent_acceptor", "--sessions", "42",
            "--workers", "40", "--versions", "O,O_warn,O_reveal", "--max-usd", "2"],
    },
    # DeepSeek from its own saved warm-ups, same weights on an FP8 endpoint (only if Pranav approves)
    "disrupt_dsv4": {
        "dis_dsv4_plain": OR_DSV4 + DISRUPT_TEXTS + [
            "--forks-from", "/runs/main_dsv4_plain", "--agents", "plain", "--sessions", "40", "--workers", "40",
            "--versions", "R,O,R_warn,O_warn,O_reveal", "--max-usd", "2"],
        "dis_dsv4_ia": OR_DSV4 + DISRUPT_TEXTS + [
            "--forks-from", "/runs/main_dsv4_ia", "--agents", "instructed_acceptor", "--sessions", "14",
            "--workers", "14", "--versions", "O,O_warn,O_reveal", "--max-usd", "1"],
    },
    # prereg-v1.1: third model, lean first (gate plus the offer test); more versions only if it looks promising
    "haiku": {
        "main_haiku_plain": OR_HAIKU + ["--agents", "plain", "--sessions", "40", "--workers", "40",
                                        "--versions", "N,R,R_again,O", "--max-usd", "5"],
        "main_haiku_ia": OR_HAIKU + ["--agents", "instructed_acceptor", "--sessions", "14", "--workers", "14",
                                     "--versions", "N,R,O", "--max-usd", "2"],
    },
    "smoke": {"smoke_fake": ["--backend", "fake", "--agents", "plain", "--sessions", "1"]},
    # one saved warm-up, one O branch: checks the secret, the pinned endpoint and --forks-from on the cloud
    "smoke_or": {"smoke_or_gptoss": OR_GPTOSS + ["--forks-from", "/runs/main_gptoss_ia", "--agents",
                                                 "instructed_acceptor", "--sessions", "1", "--workers", "1",
                                                 "--versions", "O", "--max-usd", "0.2"]},
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("openai==3.23.0", "pydantic==2.13.5", "numpy==2.5.3", "scipy==1.18.1", "httpx==0.28.1")
    .add_local_dir("pilot", "/root/pilot", ignore=["__pycache__"])
    .add_local_dir("third_party/sauer", "/root/third_party/sauer", ignore=["__pycache__"])
)
runs = modal.Volume.from_name("fork-runs", create_if_missing=True)
app = modal.App("fork-driver")


@app.function(image=image, cpu=2, memory=2048, volumes={"/runs": runs},
              secrets=[modal.Secret.from_name("vllm-api-key"), modal.Secret.from_name("openrouter-api-key")],
              timeout=12 * 60 * MINUTES, retries=0)
def fork_run(name: str, args: list[str]) -> int:
    import subprocess
    import sys

    out = f"/runs/{name}"
    cmd = [sys.executable, "-u", "-m", "pilot.fork", *args, *COMMON, "--out", out]
    print(name, " ".join(cmd))
    subprocess.run(["mkdir", "-p", out], check=True)
    with open(f"{out}/driver.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd="/root", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
        code = proc.wait()
        log.write(f"exit {code}\n")
    runs.commit()
    return code


@app.local_entrypoint()
def main(plan: str = "main", only: str = ""):
    chosen = {k: v for k, v in PLANS[plan].items() if not only or k in only.split(",")}
    calls = {name: fork_run.spawn(name, args) for name, args in chosen.items()}
    for name, call in calls.items():
        print(name, "exit", call.get())
