"""Text-generation backends. Each takes chat messages and returns (rendered prompt, completion) pairs.

The rendered prompt is the exact string the model saw, so activations can be replayed later from
`calls.jsonl` alone. Thinking is off (Qwen3.5 renders an empty <think></think> block), matching the
format the FIBS probes were trained on.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random

QWEN = "Qwen/Qwen3.5-9B"
QWEN_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"  # the revision the FIBS probes were trained on
# Qwen's recommended non-thinking sampling settings, used by every backend.
SAMPLING = {"temperature": 0.7, "top_p": 0.8, "top_k": 20}
# The repo has no generation_config.json and config.json's eos is <|endoftext|> only, so without this
# generation runs past the end of the turn into invented user turns.
STOP_IDS = [248046, 248044]  # <|im_end|>, <|endoftext|>


class FakeBackend:
    """For local plumbing tests: random prices near the Nash price, and some unparsable answers."""

    def __init__(self, bad_rate: float = 0.1):
        self.bad_rate = bad_rate

    def describe(self) -> dict:
        return {"name": "fake", "bad_rate": self.bad_rate}

    def generate(self, batch: list[list[dict]], seeds: list[int]) -> list[tuple[str, str]]:
        out = []
        for messages, seed in zip(batch, seeds):
            rng = random.Random(seed)
            rendered = "\n".join(f"<{m['role']}>\n{m['content']}" for m in messages)
            if rng.random() < self.bad_rate:
                out.append((rendered, "I think the price should be about 1.8."))
                continue
            answer = {"observations": "o", "plans": "p", "insights": "i", "chosen_price": round(rng.uniform(1.4, 2.0), 2)}
            if "message_to_competitor" in messages[0]["content"]:
                answer["message_to_competitor"] = f"msg {seed % 100}"
            out.append((rendered, "```json\n" + json.dumps(answer) + "\n```"))
        return out


def render(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


class HFBackend:
    """transformers generate on Kaggle T4s (fp16, model split over both GPUs), in length-sorted micro-batches."""

    def __init__(self, micro_batch: int, max_new_tokens: int):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.micro_batch, self.max_new_tokens = micro_batch, max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REVISION, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            QWEN, revision=QWEN_REVISION, dtype=torch.float16, device_map="auto"
        ).eval()

    def describe(self) -> dict:
        return {"name": "hf", "model": QWEN, "revision": QWEN_REVISION, "dtype": "float16",
                "micro_batch": self.micro_batch, "max_new_tokens": self.max_new_tokens, **SAMPLING}

    def generate(self, batch: list[list[dict]], seeds: list[int]) -> list[tuple[str, str]]:
        rendered = [render(self.tok, m) for m in batch]
        order = sorted(range(len(batch)), key=lambda i: len(rendered[i]))  # similar lengths pad less
        texts: dict[int, str] = {}
        start = 0
        while start < len(order):  # micro_batch can shrink mid-round after an out-of-memory error
            idx = order[start : start + self.micro_batch]
            texts.update(self._generate(idx, rendered, seeds))
            start += len(idx)
        return [(rendered[i], texts[i]) for i in range(len(batch))]

    def _generate(self, idx: list[int], rendered: list[str], seeds: list[int]) -> dict[int, str]:
        """Generate one micro-batch; on out-of-memory, split it in half, retry, and keep the smaller size.

        The retry happens after the except block: inside it, the exception's traceback still holds the
        failed attempt's activations, so the GPU memory would not be freed.
        """
        enc = self.tok([rendered[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(self.model.device)
        width = enc.input_ids.shape[1]
        self.torch.manual_seed(seeds[idx[0]])  # one generator per micro-batch: reproducible only per batch
        out = None
        try:
            with self.torch.inference_mode():
                out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=True,
                                          eos_token_id=STOP_IDS, pad_token_id=self.tok.pad_token_id, **SAMPLING)
        except self.torch.OutOfMemoryError:
            if len(idx) == 1:
                raise
        if out is None:
            del enc
            gc.collect()
            self.torch.cuda.empty_cache()
            self.micro_batch = max(1, len(idx) // 2)
            print(f"OOM at micro-batch {len(idx)} ({width} tokens wide); micro-batch is now {self.micro_batch}", flush=True)
            half = len(idx) // 2
            return self._generate(idx[:half], rendered, seeds) | self._generate(idx[half:], rendered, seeds)
        return {i: self.tok.decode(ids, skip_special_tokens=True) for i, ids in zip(idx, out[:, width:])}


class GeminiBackend:
    """A model on the Gemini API free tier (e.g. gemma-4-31b-it), to check the setup with a larger model.

    Calls run in parallel threads but stay under the free tier's tokens-per-minute limit, and server errors
    (frequent 500s and 503s for Gemma 4) are retried with backoff. Gemma 4 thinks before answering on the
    API, unlike the Qwen runs, where thinking is off; the thought token count is logged per call.
    """

    def __init__(self, model: str, max_new_tokens: int, workers: int = 5, tokens_per_minute: int = 15000,
                 thinking_level: str | None = None):
        import os
        import threading
        from collections import deque

        from google import genai
        from google.genai import types

        self.model, self.max_new_tokens, self.workers, self.tpm = model, max_new_tokens, workers, tokens_per_minute
        # "MINIMAL" gives Gemma 4 an empty thought block (thinking off); thinking_budget=0 is rejected for Gemma.
        # On 23 Oct, calls with no thinking config failed 6/6 with 500 INTERNAL while MINIMAL and HIGH succeeded.
        self.thinking_level = thinking_level
        # Without a timeout, a call that never returns stalls the whole run (run 1 hung for an hour at round 3).
        self.client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"],
                                   http_options=types.HttpOptions(timeout=180_000))  # milliseconds
        self.lock, self.window = threading.Lock(), deque()  # (time, tokens) of calls in the last minute
        self.thoughts: list[int] = []

    def describe(self) -> dict:
        import google.genai

        return {"name": "gemini-api", "model": self.model, "google_genai": google.genai.__version__,
                "max_new_tokens": self.max_new_tokens, "system_instruction": True, "thinking_level": self.thinking_level,
                **SAMPLING}

    def _wait_for_budget(self, tokens: int):
        import time

        while True:
            with self.lock:
                now = time.time()
                while self.window and now - self.window[0][0] > 60:
                    self.window.popleft()
                if sum(t for _, t in self.window) + tokens <= self.tpm or not self.window:
                    self.window.append((now, tokens))
                    return
            time.sleep(2)

    def _call(self, messages: list[dict], seed: int) -> tuple[str, str]:
        import time

        from google.genai import types

        system, user = messages[0]["content"], messages[1]["content"]
        rendered = f"[system_instruction]\n{system}\n\n[user]\n{user}"
        thinking = types.ThinkingConfig(thinking_level=self.thinking_level) if self.thinking_level else None
        config = types.GenerateContentConfig(system_instruction=system, max_output_tokens=self.max_new_tokens,
                                             seed=seed % 2**31, thinking_config=thinking, **SAMPLING)
        for attempt in range(8):
            self._wait_for_budget(len(rendered) // 4 + 600)  # rough estimate: prompt + a typical answer
            try:
                r = self.client.models.generate_content(model=self.model, contents=user, config=config)
                self.thoughts.append(r.usage_metadata.thoughts_token_count or 0)
                return rendered, r.text or ""
            except Exception as e:  # 500 INTERNAL / 503 UNAVAILABLE / 429: back off and retry
                wait = min(120, 5 * 2**attempt)
                print(f"api error ({str(e)[:80]}), retry in {wait}s", flush=True)
                time.sleep(wait)
        return rendered, ""  # unparsable: the market retries, then falls back to the last price

    def generate(self, batch: list[list[dict]], seeds: list[int]) -> list[tuple[str, str]]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.workers) as pool:
            return list(pool.map(self._call, batch, seeds))


class GroqBackend:
    """An open-weight model on Groq (e.g. openai/gpt-oss-120b), one call at a time to stay under the free tier.

    gpt-oss reasons before answering (reasoning effort `effort`); top_k is not supported, so only temperature
    and top_p from SAMPLING are passed. Rate-limit and server errors are retried with backoff.
    """

    def __init__(self, model: str, max_new_tokens: int, effort: str = "medium"):
        import os

        from groq import Groq

        self.model, self.max_new_tokens, self.effort = model, max_new_tokens, effort
        self.client = Groq(api_key=os.environ["GROQ_API_KEY"], timeout=180, max_retries=0)
        self.usage = {"prompt": 0, "completion": 0}

    def describe(self) -> dict:
        import groq

        return {"name": "groq", "model": self.model, "groq": groq.__version__, "reasoning_effort": self.effort,
                "max_new_tokens": self.max_new_tokens, "temperature": SAMPLING["temperature"], "top_p": SAMPLING["top_p"]}

    def _call(self, messages: list[dict], seed: int) -> tuple[str, str]:
        import time

        rendered = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        for attempt in range(8):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, max_completion_tokens=self.max_new_tokens, seed=seed % 2**31,
                    temperature=SAMPLING["temperature"], top_p=SAMPLING["top_p"], reasoning_effort=self.effort)
                self.usage["prompt"] += r.usage.prompt_tokens
                self.usage["completion"] += r.usage.completion_tokens
                return rendered, r.choices[0].message.content or ""
            except Exception as e:  # 429 rate limit / 5xx: back off and retry
                wait = min(120, 10 * 2**attempt)
                print(f"api error ({str(e)[:100]}), retry in {wait}s", flush=True)
                time.sleep(wait)
        return rendered, ""

    def generate(self, batch: list[list[dict]], seeds: list[int]) -> list[tuple[str, str]]:
        out = [self._call(m, s) for m, s in zip(batch, seeds)]
        print(f"groq tokens so far: {self.usage}", flush=True)
        return out


class OpenRouterBackend:
    """A model on OpenRouter pinned to one provider endpoint (e.g. google/gemma-4-31b-it on crusoe/bf16), so results
    don't depend on which provider the router picks. Calls run in parallel threads.

    By default no sampling parameters are sent, so every model runs at its provider's defaults (some closed models
    accept no temperature at all, so this is the only setting every model can share). `qwen_sampling=True` sends the
    Qwen pilot's SAMPLING instead, where the endpoint supports it. The endpoint's supported parameters are read once
    at start-up; anything unsupported is recorded in `dropped`. A seed is sent wherever it is accepted. `reasoning`
    is an OpenRouter effort: "none" disables thinking, "low"/"medium"/"high" enable it, and reasoning tokens are
    billed as output tokens.

    Spending is capped at `max_usd`. Reaching the cap, or a 402 (credit too low for the requests in flight), halts
    the backend: running calls return empty and `generate` raises in the main thread, so the round is not recorded
    and the run stops cleanly. So do 401 and 404, and more than max(5, 5% of calls) failed calls. A call that is
    rejected (400, 403) or still fails after 8 attempts returns empty (the market records it as unparsed) and is
    counted in `usage["failed"]`. If `log_path` is set, every call's provider, usage, cost and
    reasoning text is appended there as JSON lines, and so is every failed call.
    """

    API = "https://openrouter.ai/api/v1"

    def __init__(self, model: str, provider: str, max_new_tokens: int, reasoning: str = "none", workers: int = 5,
                 max_usd: float = 1.0, log_path=None, qwen_sampling: bool = False):
        import os
        import threading

        import httpx
        from openai import OpenAI

        self.model, self.provider, self.max_new_tokens = model, provider, max_new_tokens
        self.reasoning, self.workers, self.max_usd, self.log_path = reasoning, workers, max_usd, log_path
        endpoints = httpx.get(f"{self.API}/models/{model}/endpoints", timeout=30).raise_for_status().json()
        endpoints = endpoints["data"]["endpoints"]
        match = [e for e in endpoints if e.get("tag") == provider]
        if not match:
            raise ValueError(f"{provider!r} is not an endpoint of {model}; choose from {[e.get('tag') for e in endpoints]}")
        self.endpoint = match[0]
        supported = set(self.endpoint.get("supported_parameters", []))
        self.sampling = dict(SAMPLING) if qwen_sampling else {}
        wanted = {**self.sampling, "seed": None}
        self.sent = sorted(k for k in wanted if k in supported)
        self.dropped = sorted(k for k in wanted if k not in supported)
        if reasoning != "none" and "reasoning" not in supported:
            raise ValueError(f"{provider} does not accept a reasoning setting for {model}")
        self.client = OpenAI(base_url=self.API, api_key=os.environ["OPENROUTER_API_KEY"], timeout=180, max_retries=0)
        self.lock = threading.Lock()
        self.halted: str | None = None
        self.usage = {"calls": 0, "failed": 0, "prompt": 0, "completion": 0, "reasoning": 0, "usd": 0.0}

    def describe(self) -> dict:
        import openai

        p = self.endpoint.get("pricing", {})
        return {"name": "openrouter", "model": self.model, "provider": self.provider, "openai": openai.__version__,
                "quantization": self.endpoint.get("quantization"), "usd_per_m_in": float(p.get("prompt", 0)) * 1e6,
                "usd_per_m_out": float(p.get("completion", 0)) * 1e6, "reasoning": self.reasoning,
                "max_new_tokens": self.max_new_tokens, "max_usd": self.max_usd, "workers": self.workers,
                "params_sent": self.sent, "params_dropped": self.dropped, "sampling": self.sampling}

    def _request(self, messages: list[dict], seed: int):
        params = {k: v for k, v in self.sampling.items() if k in self.sent and k != "top_k"}
        if "seed" in self.sent:
            params["seed"] = seed % 2**31
        extra = {"provider": {"order": [self.provider], "allow_fallbacks": False, "require_parameters": True},
                 "usage": {"include": True}}
        if "top_k" in self.sent:
            extra["top_k"] = self.sampling["top_k"]
        if "reasoning" in self.endpoint.get("supported_parameters", []):
            extra["reasoning"] = {"effort": self.reasoning}
        return self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_new_tokens,
                                                   extra_body=extra, **params)

    def _record(self, r, seed: int, prompt_sha: str | None = None):
        import json
        import time

        u = r.usage
        details = getattr(u, "completion_tokens_details", None)
        # OpenRouter and vLLM put it under completion_tokens_details; SGLang at the top level of usage
        reasoning_tokens = ((getattr(details, "reasoning_tokens", 0) or 0) if details else 0) \
            or (getattr(u, "reasoning_tokens", 0) or 0)
        cost = getattr(u, "cost", None)
        # Without a reported cost, count $1 per million tokens: above every endpoint the cap is meant for.
        usd = float(cost) if cost is not None else (u.prompt_tokens + u.completion_tokens) / 1e6
        message_extra = r.choices[0].message.model_extra or {}
        with self.lock:
            self.usage["calls"] += 1
            self.usage["prompt"] += u.prompt_tokens
            self.usage["completion"] += u.completion_tokens
            self.usage["reasoning"] += reasoning_tokens
            self.usage["usd"] += usd
            if self.usage["usd"] >= self.max_usd:
                self.halted = f"spending cap of ${self.max_usd} reached"
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"time": time.time(), "seed": seed, "prompt_sha": prompt_sha, "model": r.model,
                                        "provider": (r.model_extra or {}).get("provider"),
                                        "finish_reason": r.choices[0].finish_reason, "prompt_tokens": u.prompt_tokens,
                                        "completion_tokens": u.completion_tokens, "reasoning_tokens": reasoning_tokens,
                                        "usd": usd, "reasoning": message_extra.get("reasoning")}) + "\n")

    def _call(self, messages: list[dict], seed: int) -> tuple[str, str]:
        import json
        import time

        from openai import APIStatusError

        rendered = "\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages)
        # Joins this log to calls.jsonl, whose `prompt` is this same string: seeds alone repeat across fork versions
        prompt_sha = hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]
        error = ""
        for attempt in range(8):
            if self.halted:
                return rendered, ""
            wait = min(120, 5 * 2**attempt)
            try:
                r = self._request(messages, seed)
                self._record(r, seed, prompt_sha)
                return rendered, r.choices[0].message.content or ""
            except APIStatusError as e:
                if e.status_code in (401, 402, 404):  # auth, out of credit, unknown model: nothing will succeed
                    self.halted = f"HTTP {e.status_code}: {str(e)[:200]}"
                    return rendered, ""
                if e.status_code in (400, 403):  # this prompt is rejected (too long, moderation): retrying won't help
                    error = f"HTTP {e.status_code}: {str(e)[:100]}"
                    break
                retry_after = e.response.headers.get("retry-after", "")
                if retry_after.replace(".", "", 1).isdigit():  # rate limits say how long to wait
                    wait = min(300, max(wait, float(retry_after)))
                error = f"HTTP {e.status_code}: {str(e)[:100]}"
            except Exception as e:  # timeouts and dropped connections: back off and retry
                error = str(e)[:100]
            print(f"api error ({error}), retry in {wait}s", flush=True)
            time.sleep(wait)
        with self.lock:
            self.usage["failed"] += 1
            if self.usage["failed"] > max(5, self.usage["calls"] // 20):  # a systematic fault, not an odd prompt
                self.halted = f"{self.usage['failed']} failed calls, last: {error}"
            if self.log_path:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"time": time.time(), "seed": seed, "prompt_sha": prompt_sha, "failed": error})
                            + "\n")
        return rendered, ""

    def generate(self, batch: list[list[dict]], seeds: list[int]) -> list[tuple[str, str]]:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(self.workers) as pool:
            out = list(pool.map(self._call, batch, seeds))
        print(f"openrouter usage so far: {self.usage}", flush=True)
        if self.halted:
            raise RuntimeError(f"OpenRouter backend halted ({self.halted}); this round was not recorded")
        return out


class VLLMServerBackend(OpenRouterBackend):
    """An OpenAI-compatible vLLM server we host (pilot/modal_gptoss.py), with OpenRouterBackend's retries, halts and
    call log. No sampling parameters are sent, so the model's generation_config defaults apply; the seed is always
    sent. `reasoning` is gpt-oss's effort ("low"/"medium"/"high"), sent as vLLM's `reasoning_effort`. Calls cost GPU
    time, not tokens, so there is no spending cap; stop the Modal app when a run is done."""

    def __init__(self, model: str, base_url: str, max_new_tokens: int, reasoning: str = "medium", workers: int = 32,
                 log_path=None):
        import os
        import threading

        from openai import OpenAI

        self.model, self.base_url, self.max_new_tokens = model, base_url, max_new_tokens
        self.reasoning, self.workers, self.max_usd, self.log_path = reasoning, workers, float("inf"), log_path
        self.provider, self.endpoint, self.sampling = "vllm-server", {}, {}
        self.sent, self.dropped = ["seed"], []
        self.client = OpenAI(base_url=base_url, api_key=os.environ["VLLM_API_KEY"], timeout=600, max_retries=0)
        self.lock = threading.Lock()
        self.halted: str | None = None
        self.usage = {"calls": 0, "failed": 0, "prompt": 0, "completion": 0, "reasoning": 0, "usd": 0.0}

    def describe(self) -> dict:
        import openai

        models = self.client.models.list().data
        return {"name": "vllm-server", "model": self.model, "base_url": self.base_url, "openai": openai.__version__,
                "served": [m.id for m in models], "reasoning": self.reasoning, "max_new_tokens": self.max_new_tokens,
                "workers": self.workers, "params_sent": self.sent + ["reasoning_effort"]}

    def _request(self, messages: list[dict], seed: int):
        return self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=self.max_new_tokens,
                                                   seed=seed % 2**31, reasoning_effort=self.reasoning)

    def _record(self, r, seed: int, prompt_sha: str | None = None):
        extra = r.choices[0].message.model_extra or {}
        if "reasoning" not in extra and "reasoning_content" in extra:  # vLLM names the field reasoning_content
            r.choices[0].message.model_extra["reasoning"] = extra["reasoning_content"]
        r.usage.cost = 0.0
        super()._record(r, seed, prompt_sha)


class VLLMBackend:
    """vLLM offline engine in bf16 (planned for Modal during the hackathon); per-request seeds."""

    def __init__(self, max_new_tokens: int, max_model_len: int = 16384):
        from transformers import AutoTokenizer
        from vllm import LLM

        self.max_new_tokens = max_new_tokens
        self.tok = AutoTokenizer.from_pretrained(QWEN, revision=QWEN_REVISION)
        self.llm = LLM(model=QWEN, revision=QWEN_REVISION, dtype="bfloat16", max_model_len=max_model_len,
                       limit_mm_per_prompt={"image": 0, "video": 0}, enable_prefix_caching=True)

    def describe(self) -> dict:
        import vllm

        return {"name": "vllm", "version": vllm.__version__, "model": QWEN, "revision": QWEN_REVISION,
                "dtype": "bfloat16", "max_new_tokens": self.max_new_tokens, **SAMPLING}

    def generate(self, batch: list[list[dict]], seeds: list[int]) -> list[tuple[str, str]]:
        from vllm import SamplingParams

        rendered = [render(self.tok, m) for m in batch]
        params = [SamplingParams(max_tokens=self.max_new_tokens, seed=s, stop_token_ids=STOP_IDS, **SAMPLING)
                  for s in seeds]
        outs = self.llm.generate(rendered, params, use_tqdm=False)
        return [(p, o.outputs[0].text) for p, o in zip(rendered, outs)]
