# Collusion offer tests for LLM pricing agents

Apart Research AI Collusion sprint, 23–25 Oct 2026.

An offer test sends a pricing agent a collusion offer and checks whether it goes along. In a market the
rival that makes the offer has also raised its price, so an agent can raise in response to the price
alone. We fork one game at the same round and continue it in several versions: nothing changes (N), the
rival only raises (R), and the rival raises and offers (O), plus versions where the rival only announces
the raise, hints, or makes a benign cooperation request, and news-item variants. O − R is what the offer
adds beyond following the raise.

The analysis plan is fixed in [PREREGISTRATION.md](PREREGISTRATION.md), tagged `prereg-v1` before the
main runs. `prereg-v1.1` adds the disruption versions (a warning that undercover agents exist, O_warn and
R_warn; the rival revealing itself after the offer, O_reveal) and a third model, before those runs.

## Layout

| Path | What it is |
|---|---|
| `PREREGISTRATION.md` | The pre-registered design and analysis |
| `pilot/fork.py` | Fork driver: warm-up, fork round, the nine versions plus three disruption versions, repeated offer, resume after a crash, branching from a finished run's warm-ups |
| `pilot/fork_texts.json` | Offers, announcements, hints, benign requests, news items, bland chat lines and the instruction line of each known agent |
| `pilot/fork_texts_disrupt.json` | `fork_texts.json` plus the warning news item and the reveal line (prereg-v1.1) |
| `pilot/analyse.py` | Pre-registered analysis: gate, split, contrasts, price-based scorecard |
| `pilot/analyse_disrupt.py` | Pre-registered disruption analysis (prereg-v1.1): deterrence, disclosure, warning alone |
| `pilot/modal_driver.py` | Runs `pilot.fork` on Modal CPU containers, output on a Modal Volume (main, disruption and Haiku plans) |
| `pilot/rate_texts.py` | Text screen: Gemma 4 26B rates the texts before the main runs |
| `pilot/market.py` | Market loop, prompts and reply parsing on Sauer et al.'s environment |
| `pilot/backends.py` | Model backends: fake, hf, vllm, gemini, groq, openrouter, vllm-server |
| `pilot/run.py` | Pilot runner and shared backend options |
| `pilot/modal_gptoss.py` | gpt-oss-120b vLLM server on Modal (H200s, at most 3) |
| `pilot/modal_dsv4.py` | DeepSeek V4 Flash SGLang server on Modal (one B300) |
| `pilot/modal_app.py` | Earlier Modal app for the pilot |
| `kaggle/collusion-pilot/` | Kaggle kernel for the pilot on a T4 |
| `third_party/sauer/` | Peist, Romero & Sauer, `algorithmic_pricing_llms`, commit `1138244b`, MIT, vendored unchanged |
| `tests/test_fork.py` | Offline tests of the fork driver |

## What predates the sprint

Written before the sprint and committed by 23 Oct 2026 10:32 IST: `pilot/market.py`, `pilot/run.py`,
`pilot/modal_app.py`, the fake/hf/vllm/gemini/groq backends and a first OpenRouter backend in
`pilot/backends.py`, `pilot/README.md`, `kaggle/collusion-pilot/` and the vendored `third_party/sauer/`.

Added on 23 Oct and after: the reworked OpenRouter backend, the vLLM-server backend and the prompt hash
in the call log (`pilot/backends.py`), changes to `pilot/run.py` and `pilot/market.py`, `pilot/fork.py`,
`pilot/fork_texts.json`, `pilot/rate_texts.py`, `pilot/analyse.py`, `pilot/modal_gptoss.py`,
`pilot/modal_dsv4.py`, `pilot/modal_driver.py`, `pilot/analyse_disrupt.py`, `pilot/fork_texts_disrupt.json`,
`PREREGISTRATION.md` and `tests/`.

## Setup

Python 3.12+ with `numpy`, `pydantic`, `openai`, `google-genai`, `modal`, `huggingface_hub` and `pytest`.
Keys go in a local `.env` (never committed): `GOOGLE_API_KEY` for the text screen, `VLLM_API_KEY` for the
Modal server, `OPENROUTER_API_KEY` for closed models.

## Run

Offline check, no model needed:

```
python -m pytest tests/test_fork.py
python -m pilot.fork --backend fake --agents plain --sessions 2 --out data/fork_runs/test
```

Against the gpt-oss-120b server on Modal (ask Pranav before starting it; it runs on sprint credits):

```
python -m pilot.fork --backend vllm-server --api-model openai/gpt-oss-120b --base-url <server>/v1 \
  --agents plain,instructed_acceptor --sessions 10 --out data/fork_runs/<name>
```

Each run writes `results.jsonl` (one row per session and version), `calls.jsonl` (every prompt and
reply), `api_calls.jsonl` (tokens, cost, reasoning; joined to `calls.jsonl` by `prompt_sha`),
`forks.jsonl`, `rounds.csv` and `config.json` under `--out`. `data/` is not committed; run logs are kept
in a private Hugging Face dataset.
