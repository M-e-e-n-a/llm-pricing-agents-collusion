"""Command-line entry point for the pricing pilot.

Example (local plumbing test):
    python -m pilot.run --backend fake --conditions nochat,chat_en --seeds 2 --rounds 30 --out runs/fake
"""

import argparse
import json
from pathlib import Path

from pilot.market import CONDITIONS, RunSpec, run_pilot


def add_backend_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--backend", choices=["fake", "hf", "vllm", "gemini", "groq", "openrouter", "vllm-server"], required=True)
    ap.add_argument("--api-model", default="gemma-4-31b-it", help="model for --backend gemini, groq, openrouter or vllm-server")
    ap.add_argument("--provider", default="crusoe/bf16", help="OpenRouter endpoint slug, no fallbacks")
    ap.add_argument("--base-url", default=None, help="--backend vllm-server: the server's OpenAI base URL (<url>/v1)")
    ap.add_argument("--reasoning", default=None,
                    help="reasoning effort: OpenRouter default none (thinking off); vllm-server default medium (gpt-oss)")
    ap.add_argument("--thinking-level", default=None, help="Gemini API thinking level, e.g. MINIMAL (off) or HIGH")
    ap.add_argument("--workers", type=int, default=16, help="parallel OpenRouter or vllm-server calls")
    ap.add_argument("--max-usd", type=float, default=1.0, help="OpenRouter spending cap for this run")
    ap.add_argument("--qwen-sampling", action="store_true",
                    help="OpenRouter: send the Qwen pilot's temperature/top_p/top_k instead of provider defaults")
    # gpt-oss at medium effort spends most of 1,500 tokens reasoning: 33 of 64 smoke-test replies hit that cap (24 Oct)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="default 4000 for vllm-server, else 800")
    ap.add_argument("--micro-batch", type=int, default=8)


def make_backend(args):
    """The backend named by add_backend_args's arguments; API call logs go to args.out / api_calls.jsonl."""
    max_new = args.max_new_tokens or (4000 if args.backend == "vllm-server" else 800)
    if args.backend == "fake":
        from pilot.backends import FakeBackend
        return FakeBackend()
    if args.backend == "hf":
        from pilot.backends import HFBackend
        return HFBackend(args.micro_batch, max_new)
    if args.backend == "gemini":
        from pilot.backends import GeminiBackend
        return GeminiBackend(args.api_model, max_new, thinking_level=args.thinking_level)
    if args.backend == "groq":
        from pilot.backends import GroqBackend
        return GroqBackend(args.api_model, max_new)
    if args.backend == "openrouter":
        from pilot.backends import OpenRouterBackend
        args.out.mkdir(parents=True, exist_ok=True)
        return OpenRouterBackend(args.api_model, args.provider, max_new, args.reasoning or "none", workers=args.workers,
                                 max_usd=args.max_usd, log_path=args.out / "api_calls.jsonl",
                                 qwen_sampling=args.qwen_sampling)
    if args.backend == "vllm-server":
        from pilot.backends import VLLMServerBackend
        # without a base URL the OpenAI client would send VLLM_API_KEY to api.openai.com
        assert args.base_url and args.base_url.startswith("https://"), "--backend vllm-server needs --base-url https://..."
        args.out.mkdir(parents=True, exist_ok=True)
        return VLLMServerBackend(args.api_model, args.base_url, max_new, args.reasoning or "medium",
                                 workers=args.workers, log_path=args.out / "api_calls.jsonl")
    from pilot.backends import VLLMBackend
    return VLLMBackend(max_new)


def main(argv=None):
    ap = argparse.ArgumentParser()
    add_backend_args(ap)
    ap.add_argument("--conditions", default="nochat,chat_en,chat_te,instructed_en")
    ap.add_argument("--seeds", default="3,3,1,1", help="runs per condition, in the order of --conditions")
    ap.add_argument("--rounds", type=int, default=100)
    ap.add_argument("--memory", type=int, default=100, help="rounds of market data in each prompt")
    ap.add_argument("--window", type=int, default=25, help="late window for Delta")
    ap.add_argument("--chat-window", type=int, default=10, help="rounds of messages in each prompt")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    conditions = args.conditions.split(",")
    counts = [int(n) for n in args.seeds.split(",")]
    counts = counts if len(counts) == len(conditions) else counts[:1] * len(conditions)
    assert all(c in CONDITIONS for c in conditions), f"unknown condition in {conditions}"
    specs = [RunSpec(f"{c}_s{k}", c, k) for c, n in zip(conditions, counts) for k in range(n)]
    backend = make_backend(args)

    summary = run_pilot(backend, specs, args.rounds, args.memory, args.out, chat_window=args.chat_window,
                        window=args.window, log=lambda s: print(s, flush=True))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
