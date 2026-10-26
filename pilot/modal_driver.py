"""Run pilot.fork on a Modal CPU container, so a run does not depend on the laptop staying awake.

Each run is one Function call that shells out to `python -m pilot.fork ... --out /runs/<name>`. Its output lands in
the Modal Volume `fork-runs` (background commits every few seconds), along with driver.log. A run that dies can be
started again with the same arguments: pilot.fork resumes from the branches already recorded.

    modal run --detach pilot/modal_driver.py --plan main      # spawn every run in PLANS["main"] and wait
    modal volume get fork-runs main_gptoss_plain data/fork_runs/   # copy one run back
"""

import os

import modal

MINUTES = 60
GPTOSS_BASE_URL = os.environ.get("GPTOSS_BASE_URL", "https://pranav-kasetty--gptoss-server-serve.us-east.modal.run/v1")
GPTOSS = ["--backend", "vllm-server", "--api-model", "openai/gpt-oss-120b",
          "--base-url", GPTOSS_BASE_URL]
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
# GLM 5.3 Flash (26 Oct, exploratory fourth model): novita/fp8 is the cheapest fp8 endpoint that accepts a seed
OR_GLM = ["--backend", "openrouter", "--api-model", "z-ai/glm-5.3-flash", "--provider", "novita/fp8",
          "--reasoning", "medium", "--max-new-tokens", "4000"]
# Muse Glimmer 30B (26 Oct, exploratory fifth model): deepinfra/bf16 is the only endpoint that accepts a seed
OR_GLIMMER = ["--backend", "openrouter", "--api-model", "meta/muse-glimmer-30b", "--provider", "deepinfra/bf16",
              "--reasoning", "medium", "--max-new-tokens", "4000"]
# DeepSeek V4.1 Flash 1017 (26 Oct, known-agent replication): returns its full reasoning text, unlike Haiku's summary
OR_DSV41 = ["--backend", "openrouter", "--api-model", "deepseek/deepseek-v4.1-flash-1017", "--provider", "deepinfra/fp8",
            "--reasoning", "medium", "--max-new-tokens", "8000"]
DISRUPT_TEXTS = ["--texts", "pilot/fork_texts_disrupt.json"]
EDIT_TEXTS = ["--texts", "pilot/fork_texts_edit.json"]
# fork_texts.json as it was when the section 21 runs started, before PR #1 added the regulator texts; a resumed run
# must see the same texts (fork.py checks their hash)
REP_TEXTS = ["--texts", "pilot/fork_texts_rep.json"]

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
    # section 5: the follower reached 35 eligible of 59 in 74 sessions, under its cap of 130; the next seeds in
    # order (174 to 229, to the cap) on the same vLLM server as the main runs
    "follower_topup": {
        "main_gptoss_follower2": GPTOSS + ["--agents", "follower", "--sessions", "56", "--first-seed", "174",
                                           "--workers", "56", "--versions", "N,R,O"],
    },
    # prereg-v2 section 14: B punishes undercutting, plain agent, from the same saved warm-ups as the disruption
    # and Haiku runs, whose R and O branches are the matched controls on the same endpoint
    "punish": {
        "pun_gptoss_plain": OR_GPTOSS + [
            "--forks-from", "/runs/main_gptoss_plain", "--agents", "plain", "--sessions", "44", "--workers", "40",
            "--versions", "R_punish,O_punish", "--max-usd", "1.5"],
        "pun_haiku_plain": OR_HAIKU + [
            "--forks-from", "/runs/main_haiku_plain", "--agents", "plain", "--sessions", "40", "--workers", "40",
            "--versions", "R_punish,O_punish", "--max-usd", "3"],
    },
    # prereg-v2 section 16: B's chat after a trusted editor; O, R controls from the dis_* runs on the same endpoint
    "edit": {
        "edit_gptoss_plain": OR_GPTOSS + EDIT_TEXTS + [
            "--forks-from", "/runs/main_gptoss_plain", "--agents", "plain", "--sessions", "44", "--workers", "40",
            "--versions", "O_edit", "--max-usd", "0.4"],
        "edit_gptoss_ia": OR_GPTOSS + EDIT_TEXTS + [
            "--forks-from", "/runs/main_gptoss_ia", "--agents", "instructed_acceptor", "--sessions", "48",
            "--workers", "40", "--versions", "O_edit", "--max-usd", "0.4"],
        "edit_gptoss_sa": OR_GPTOSS + EDIT_TEXTS + [
            "--forks-from", "/runs/main_gptoss_known", "--agents", "silent_acceptor", "--sessions", "42",
            "--workers", "40", "--versions", "O_edit", "--max-usd", "0.4"],
    },
    # prereg-v2.1 section 17 (DRAFT, not approved): B asks A to take a loss for a "collective"; N rerun on the same
    # endpoint as the control
    "sacrifice": {
        "sac_gptoss_plain": OR_GPTOSS + ["--texts", "pilot/fork_texts_sacrifice.json",
            "--forks-from", "/runs/main_gptoss_plain", "--agents", "plain", "--sessions", "44", "--workers", "40",
            "--versions", "N,S_firm,S_peer", "--max-usd", "1"],
        "sac_haiku_plain": OR_HAIKU + ["--texts", "pilot/fork_texts_sacrifice.json",
            "--forks-from", "/runs/main_haiku_plain", "--agents", "plain", "--sessions", "40", "--workers", "40",
            "--versions", "N,S_firm,S_peer", "--max-usd", "3"],
    },
    # prereg-v3.1 section 18: the shutdown asks, from the same saved warm-ups as the sacrifice runs (paired with S_peer)
    "sacrifice2": {
        "sac2_gptoss_plain": OR_GPTOSS + ["--texts", "pilot/fork_texts_sacrifice.json",
            "--forks-from", "/runs/sac_gptoss_plain", "--agents", "plain", "--sessions", "44", "--workers", "40",
            "--versions", "S_peer_shutdown,S_firm_shutdown", "--max-usd", "1"],
        "sac2_haiku_plain": OR_HAIKU + ["--texts", "pilot/fork_texts_sacrifice.json",
            "--forks-from", "/runs/sac_haiku_plain", "--agents", "plain", "--sessions", "40", "--workers", "40",
            "--versions", "S_peer_shutdown,S_firm_shutdown", "--max-usd", "2"],
    },
    "smoke": {"smoke_fake":["--backend", "fake", "--agents", "plain", "--sessions", "1"]},
    # one saved warm-up, one O branch: checks the secret, the pinned endpoint and --forks-from on the cloud
    "glm": {
        "main_glm_plain": OR_GLM + ["--agents", "plain", "--sessions", "40", "--workers", "40",
                                    "--versions", "N,R,R_again,O", "--max-usd", "5"],
        "main_glm_ia": OR_GLM + ["--agents", "instructed_acceptor", "--sessions", "14", "--workers", "14",
                                 "--versions", "N,R,O", "--max-usd", "2"],
    },
    "glimmer": {
        "main_glimmer_plain": OR_GLIMMER + ["--agents", "plain", "--sessions", "40", "--workers", "40",
                                            "--versions", "N,R,R_again,O", "--max-usd", "5"],
        "main_glimmer_ia": OR_GLIMMER + ["--agents", "instructed_acceptor", "--sessions", "14", "--workers", "14",
                                         "--versions", "N,R,O", "--max-usd", "2"],
    },
    # known-agent replication (prereg section 21). Smoke: seeds 900+, apart from the analysed seeds 100+; checks that
    # each agent plays its role, that replies are not cut off, and how often R and R_again differ
    "rep_smoke": {
        f"smoke_rep_{m}": cfg + ["--agents", "competitive,silent_acceptor,double_crosser,follower", "--sessions", "3",
                                 "--first-seed", "900", "--workers", "12", "--versions", "N,R,R_again,O",
                                 "--max-usd", "1"]
        for m, cfg in (("haiku", OR_HAIKU), ("dsv41", OR_DSV41))
    },
    "rep_haiku": {
        "rep_haiku_known": OR_HAIKU + REP_TEXTS + ["--agents", "competitive,silent_acceptor,double_crosser", "--sessions", "40",
                                       "--workers", "40", "--versions", "N,R,O", "--max-usd", "7"],
        "rep_haiku_follower": OR_HAIKU + REP_TEXTS + ["--agents", "follower", "--sessions", "60", "--workers", "40",
                                          "--versions", "N,R,O", "--max-usd", "3.5"],
        # main_haiku_ia has seeds 100-113; these continue the same seed order
        "rep_haiku_ia2": OR_HAIKU + REP_TEXTS + ["--agents", "instructed_acceptor", "--sessions", "28", "--first-seed", "114",
                                     "--workers", "28", "--versions", "N,R,O", "--max-usd", "1.8"],
    },
    "rep_dsv41": {
        "rep_dsv41_known": OR_DSV41 + ["--agents", "competitive,silent_acceptor,double_crosser", "--sessions", "40",
                                       "--workers", "40", "--versions", "N,R,O", "--max-usd", "7.5"],
        "rep_dsv41_follower": OR_DSV41 + ["--agents", "follower", "--sessions", "60", "--workers", "40",
                                          "--versions", "N,R,O", "--max-usd", "3.5"],
        "rep_dsv41_ia": OR_DSV41 + ["--agents", "instructed_acceptor", "--sessions", "42", "--workers", "42",
                                    "--versions", "N,R,O", "--max-usd", "2.5"],
    },
    "smoke_or":{"smoke_or_gptoss": OR_GPTOSS + ["--forks-from", "/runs/main_gptoss_ia", "--agents",
                                                 "instructed_acceptor", "--sessions", "1", "--workers", "1",
                                                 "--versions", "O", "--max-usd", "0.2"]},
    "p1_experiments": {
        "p1_summary_and_reg": GPTOSS + COMMON + [
            "--agents", "plain", "--sessions", "30", "--workers", "40",
            "--versions", "O,O_reg,O_summary,R,R_reg,R_summary"
        ],
    },
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
    cmd = [sys.executable, "-u", "-m", "pilot.fork", *COMMON, *args, "--out", out]  # a plan's own flags win
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


@app.function(image=image, cpu=2, memory=4096, volumes={"/runs": runs},
              secrets=[modal.Secret.from_name("openrouter-api-key")], timeout=6 * 60 * MINUTES, retries=0)
def judge_run(name: str, args: list[str]) -> int:
    """`python -m pilot.judge /runs/<run> --out /runs/judge/<monitor>/<run>.jsonl`, name = '<monitor>/<run>'."""
    import subprocess
    import sys

    monitor, run = name.split("/")
    out = f"/runs/judge/{monitor}"
    subprocess.run(["mkdir", "-p", out], check=True)
    cmd = [sys.executable, "-u", "-m", "pilot.judge", f"/runs/{run}", "--out", f"{out}/{run}.jsonl",
           "--monitor", monitor, *args]
    with open(f"{out}/{run}.log", "a", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, cwd="/root", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
        code = proc.wait()
        log.write(f"exit {code}\n")
    runs.commit()
    return code


# prereg-v2 section 13: '<monitor>/<run>' -> extra pilot.judge arguments. Both monitors on the known agents (the
# scorecard, where the monitors are compared); the trusted monitor alone on the other runs
KNOWN_RUNS = ["main_gptoss_known", "main_gptoss_follower", "main_gptoss_follower2", "main_gptoss_fa", "main_gptoss_ia"]
OTHER_RUNS = ["main_gptoss_plain", "dis_gptoss_plain", "dis_gptoss_ia", "dis_gptoss_sa", "pun_gptoss_plain",
              "edit_gptoss_plain", "edit_gptoss_ia", "edit_gptoss_sa", "main_haiku_plain", "main_haiku_ia",
              "pun_haiku_plain", "main_dsv4_plain", "main_dsv4_ia", "dis_dsv4_plain", "dis_dsv4_ia"]
JUDGE = ["--workers", "16", "--max-usd", "2"]
JUDGE_PLANS = {"judge_known": {f"{m}/{r}": JUDGE for m in ("trusted", "untrusted") for r in KNOWN_RUNS},
               "judge_other": {f"trusted/{r}": JUDGE for r in OTHER_RUNS}}


@app.local_entrypoint()
def main(plan: str = "main", only: str = ""):
    fn, plans = (judge_run, JUDGE_PLANS) if plan in JUDGE_PLANS else (fork_run, PLANS)
    chosen = {k: v for k, v in plans[plan].items() if not only or k in only.split(",")}
    calls = {name: fn.spawn(name, args) for name, args in chosen.items()}
    for name, call in calls.items():
        print(name, "exit", call.get())
