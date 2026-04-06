#!/usr/bin/env python3
"""
Benchmark two OpenAI-compatible inference endpoints (e.g. vLLM vs Friendli Engine)
with a fair, reproducible workload and generate a single latency-throughput frontier plot.

This version uses:
- Trio for orchestration, scheduling, concurrency, and cancellation
- aiohttp for HTTP streaming
- trio-asyncio as the bridge because aiohttp is asyncio-native

Why this benchmark:
- Both vLLM and Friendli expose OpenAI-compatible APIs, so the same request path can be used
  against both engines for a fair comparison.
- The most decision-useful serving metrics are p95 time-to-first-token (TTFT), output-token
  goodput, success rate, and billed tokens per resolved turn.
- The single chart produced is a latency-throughput frontier: x = p95 TTFT, y = output-token
  goodput. Better systems move up and left.

Example:
  python benchmark_friendli_vs_open_source.py \
    --oss-label vLLM \
    --oss-base-url http://localhost:8000/v1 \
    --oss-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --oss-api-key token-abc123 \
    --friendli-label Friendli \
    --friendli-base-url https://inference.friendli.ai/v1 \
    --friendli-model meta-llama-3.1-8b-instruct \
    --friendli-api-key "$FRIENDLI_API_KEY" \
    --tokenizer meta-llama/Meta-Llama-3.1-8B-Instruct \
    --prompt-file sample_prompts_code.jsonl \
    --request-rates 1,2,4,8,16 \
    --requests-per-rate 50 \
    --max-tokens 192 \
    --warmup-requests 5 \
    --outdir ./bench_results

Prompt JSONL format:
  {"prompt": "Find the bug in this Python function and propose a minimal patch."}

Install:
  pip install trio trio-asyncio aiohttp numpy matplotlib transformers
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import trio
import trio_asyncio

try:
    import aiohttp
except Exception as exc:  # pragma: no cover
    raise SystemExit("aiohttp is required: pip install aiohttp") from exc

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover
    raise SystemExit("transformers is required: pip install transformers") from exc


@dataclass(frozen=True)
class EngineConfig:
    label: str
    base_url: str
    model: str
    api_key: str


@dataclass(frozen=True)
class RequestSpec:
    request_id: int
    prompt: str
    send_at_s: float


@dataclass
class RequestResult:
    engine: str
    rate: float
    request_id: int
    prompt_chars: int
    input_tokens: int
    output_tokens: int
    success: bool
    http_status: int
    error: str
    dispatch_time: float
    first_token_time: Optional[float]
    end_time: float

    @property
    def billed_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def e2e_latency_s(self) -> float:
        return self.end_time - self.dispatch_time

    @property
    def ttft_s(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.dispatch_time

    @property
    def tpot_s(self) -> Optional[float]:
        if self.first_token_time is None or self.output_tokens <= 1:
            return None
        return (self.end_time - self.first_token_time) / max(self.output_tokens - 1, 1)


@dataclass
class AggregateMetrics:
    engine: str
    rate: float
    num_requests: int
    num_success: int
    success_rate: float
    wall_time_s: float
    request_throughput_rps: float
    output_goodput_toks_per_s: float
    billed_toks_per_resolved_turn: float
    p50_ttft_ms: Optional[float]
    p95_ttft_ms: Optional[float]
    p50_e2e_ms: Optional[float]
    p95_e2e_ms: Optional[float]
    p50_tpot_ms: Optional[float]
    p95_tpot_ms: Optional[float]


class TokenCounter:
    def __init__(self, tokenizer_name_or_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        self.has_chat_template = bool(getattr(self.tokenizer, "chat_template", None))

    def count_input_tokens(self, system_prompt: str, prompt: str) -> int:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        if self.has_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                toks = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors=None,
                )
                return len(toks)
            except Exception:
                pass
        return len(self.tokenizer.encode(system_prompt + "\n" + prompt, add_special_tokens=True))

    def count_output_tokens(self, text: str) -> int:
        if not text:
            return 0
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--oss-label", default="vLLM")
    p.add_argument("--oss-base-url", required=True, help="OpenAI-compatible base URL ending in /v1")
    p.add_argument("--oss-model", required=True)
    p.add_argument("--oss-api-key", default="token-abc123")
    p.add_argument("--friendli-label", default="Friendli")
    p.add_argument("--friendli-base-url", required=True, help="OpenAI-compatible base URL ending in /v1")
    p.add_argument("--friendli-model", required=True)
    p.add_argument("--friendli-api-key", required=True)
    p.add_argument("--tokenizer", required=True, help="HF tokenizer path shared by both engines")
    p.add_argument("--prompt-file", default=None, help="JSONL file with {'prompt': ...} per line")
    p.add_argument("--request-rates", default="1,2,4,8,16", help="Comma-separated offered request rates")
    p.add_argument("--requests-per-rate", type=int, default=40)
    p.add_argument("--warmup-requests", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--timeout-s", type=float, default=180.0)
    p.add_argument("--max-concurrency", type=int, default=256)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--fixed-seed-param", type=int, default=7)
    p.add_argument("--disable-seed-param", action="store_true")
    p.add_argument("--system-prompt", default="You are a precise coding assistant. Return concise, factual answers.")
    p.add_argument("--skip-warmup", action="store_true")
    p.add_argument("--outdir", default="bench_results")
    return p.parse_args()


def load_prompts(path: Optional[str], needed: int, seed: int) -> List[str]:
    if path is None:
        base = [
            "Given the Python traceback below, identify the root cause and propose the minimal patch. Traceback: TypeError: unsupported operand type(s) for +: 'NoneType' and 'int' in utils.py line 48.",
            "Review this code for a race condition and propose a minimal fix: two coroutines write to the same cache key without a lock.",
            "Explain why this SQL query is slow and propose a rewrite: SELECT * FROM events WHERE date(created_at)=CURRENT_DATE;",
            "Given a failing unit test for a Flask endpoint returning 500 instead of 404, propose the most likely code fix.",
            "Find the bug in this function and return a unified diff only: def pct(a,b): return a/b*100",
            "Propose a concise patch to handle EOFError in a CLI loop that reads input() repeatedly.",
            "Why might this Docker build be slow, and what is the single highest-impact optimization?",
            "Given a mypy error about Optional[str] passed to open(), suggest the smallest safe fix.",
            "A streaming SSE parser occasionally drops the last event. State the most likely bug and fix.",
            "A retry loop catches Exception broadly and masks KeyboardInterrupt. Propose the minimal correction.",
        ]
        rng = random.Random(seed)
        prompts: List[str] = []
        while len(prompts) < needed:
            prompts.extend(rng.sample(base, k=min(len(base), needed - len(prompts))))
        return prompts[:needed]

    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj if isinstance(obj, str) else obj["prompt"])
    if not prompts:
        raise ValueError("No prompts loaded")
    if len(prompts) >= needed:
        return prompts[:needed]
    out: List[str] = []
    i = 0
    while len(out) < needed:
        out.append(prompts[i % len(prompts)])
        i += 1
    return out


def build_schedule(num_requests: int, rate: float) -> List[float]:
    if rate <= 0:
        raise ValueError("rate must be > 0")
    return [i / rate for i in range(num_requests)]


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), p))


def extract_text_delta(obj: Dict[str, Any]) -> str:
    if not obj.get("choices"):
        return ""
    choice = obj["choices"][0]

    delta = choice.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("content")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "".join(parts)

    text = choice.get("text")
    if isinstance(text, str):
        return text

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    txt = item.get("text") or item.get("content")
                    if isinstance(txt, str):
                        parts.append(txt)
            return "".join(parts)
    return ""


async def aio_create_session(timeout_s: float, max_concurrency: int) -> aiohttp.ClientSession:
    timeout = aiohttp.ClientTimeout(total=timeout_s, connect=timeout_s, sock_connect=timeout_s, sock_read=timeout_s)
    connector = aiohttp.TCPConnector(limit=max_concurrency, limit_per_host=max_concurrency, force_close=False, enable_cleanup_closed=True)
    return aiohttp.ClientSession(timeout=timeout, connector=connector, raise_for_status=False)


async def aio_invoke_streaming_chat(
    session: aiohttp.ClientSession,
    engine: EngineConfig,
    token_counter: TokenCounter,
    request_spec: RequestSpec,
    rate: float,
    args: argparse.Namespace,
) -> RequestResult:
    url = engine.base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {engine.api_key}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    payload: Dict[str, Any] = {
        "model": engine.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": request_spec.prompt},
        ],
        "stream": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if not args.disable_seed_param:
        payload["seed"] = args.fixed_seed_param

    input_tokens = token_counter.count_input_tokens(args.system_prompt, request_spec.prompt)
    dispatch_time = time.perf_counter()
    first_token_time: Optional[float] = None
    output_chunks: List[str] = []
    status = 0
    success = False
    error = ""

    try:
        async with session.post(url, headers=headers, json=payload) as resp:
            status = resp.status
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(f"HTTP {resp.status}: {body[:500]}")

            buffer = ""
            async for raw_chunk in resp.content.iter_chunked(4096):
                buffer += raw_chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        success = True
                        break
                    try:
                        obj = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    piece = extract_text_delta(obj)
                    if piece:
                        output_chunks.append(piece)
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                if success:
                    break
    except Exception as exc:
        error = str(exc)

    end_time = time.perf_counter()
    output_tokens = token_counter.count_output_tokens("".join(output_chunks))
    return RequestResult(
        engine=engine.label,
        rate=rate,
        request_id=request_spec.request_id,
        prompt_chars=len(request_spec.prompt),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        success=success,
        http_status=status,
        error=error,
        dispatch_time=dispatch_time,
        first_token_time=first_token_time,
        end_time=end_time,
    )


async def invoke_streaming_chat(
    session: aiohttp.ClientSession,
    semaphore: trio.Semaphore,
    engine: EngineConfig,
    token_counter: TokenCounter,
    request_spec: RequestSpec,
    rate: float,
    args: argparse.Namespace,
    benchmark_t0: float,
    sink: List[RequestResult],
) -> None:
    async with semaphore:
        now_offset = time.perf_counter() - benchmark_t0
        if request_spec.send_at_s > now_offset:
            await trio.sleep(request_spec.send_at_s - now_offset)
        result = await trio_asyncio.aio_as_trio(aio_invoke_streaming_chat)(
            session,
            engine,
            token_counter,
            request_spec,
            rate,
            args,
        )
        sink.append(result)


async def warmup(
    session: aiohttp.ClientSession,
    semaphore: trio.Semaphore,
    engine: EngineConfig,
    token_counter: TokenCounter,
    prompts: List[str],
    args: argparse.Namespace,
) -> None:
    if args.skip_warmup or args.warmup_requests <= 0:
        return
    benchmark_t0 = time.perf_counter()
    sink: List[RequestResult] = []
    async with trio.open_nursery() as nursery:
        for i in range(args.warmup_requests):
            req = RequestSpec(request_id=-(i + 1), prompt=prompts[i % len(prompts)], send_at_s=0.0)
            nursery.start_soon(
                invoke_streaming_chat,
                session,
                semaphore,
                engine,
                token_counter,
                req,
                0.0,
                args,
                benchmark_t0,
                sink,
            )


async def run_rate_sweep_for_engine(
    engine: EngineConfig,
    token_counter: TokenCounter,
    prompts: List[str],
    rates: List[float],
    args: argparse.Namespace,
) -> List[RequestResult]:
    semaphore = trio.Semaphore(args.max_concurrency)
    all_results: List[RequestResult] = []

    async with trio_asyncio.open_loop():
        session = await trio_asyncio.aio_as_trio(aio_create_session)(args.timeout_s, args.max_concurrency)
        try:
            await warmup(session, semaphore, engine, token_counter, prompts, args)
            for rate in rates:
                schedule = build_schedule(args.requests_per_rate, rate)
                specs = [
                    RequestSpec(request_id=i, prompt=prompts[i % len(prompts)], send_at_s=schedule[i])
                    for i in range(args.requests_per_rate)
                ]
                batch_results: List[RequestResult] = []
                benchmark_t0 = time.perf_counter()
                async with trio.open_nursery() as nursery:
                    for req in specs:
                        nursery.start_soon(
                            invoke_streaming_chat,
                            session,
                            semaphore,
                            engine,
                            token_counter,
                            req,
                            rate,
                            args,
                            benchmark_t0,
                            batch_results,
                        )
                all_results.extend(sorted(batch_results, key=lambda r: r.request_id))
        finally:
            await trio_asyncio.aio_as_trio(session.close)()
    return all_results


def aggregate_results(results: List[RequestResult]) -> List[AggregateMetrics]:
    by_key: Dict[Tuple[str, float], List[RequestResult]] = defaultdict(list)
    for r in results:
        by_key[(r.engine, r.rate)].append(r)

    aggregates: List[AggregateMetrics] = []
    for (engine, rate), rows in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows = sorted(rows, key=lambda r: r.dispatch_time)
        successes = [r for r in rows if r.success]
        wall_time_s = max(r.end_time for r in rows) - min(r.dispatch_time for r in rows) if rows else 0.0
        ttft_ms = [r.ttft_s * 1000.0 for r in successes if r.ttft_s is not None]
        e2e_ms = [r.e2e_latency_s * 1000.0 for r in successes]
        tpot_ms = [r.tpot_s * 1000.0 for r in successes if r.tpot_s is not None]
        output_toks = sum(r.output_tokens for r in successes)
        billed_toks = sum(r.billed_tokens for r in successes)
        num_success = len(successes)
        aggregates.append(
            AggregateMetrics(
                engine=engine,
                rate=rate,
                num_requests=len(rows),
                num_success=num_success,
                success_rate=(num_success / len(rows)) if rows else 0.0,
                wall_time_s=wall_time_s,
                request_throughput_rps=(num_success / wall_time_s) if wall_time_s > 0 else 0.0,
                output_goodput_toks_per_s=(output_toks / wall_time_s) if wall_time_s > 0 else 0.0,
                billed_toks_per_resolved_turn=(billed_toks / num_success) if num_success else math.nan,
                p50_ttft_ms=percentile(ttft_ms, 50),
                p95_ttft_ms=percentile(ttft_ms, 95),
                p50_e2e_ms=percentile(e2e_ms, 50),
                p95_e2e_ms=percentile(e2e_ms, 95),
                p50_tpot_ms=percentile(tpot_ms, 50),
                p95_tpot_ms=percentile(tpot_ms, 95),
            )
        )
    return aggregates


def write_csv_results(path: Path, results: List[RequestResult]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "engine",
            "rate",
            "request_id",
            "success",
            "http_status",
            "prompt_chars",
            "input_tokens",
            "output_tokens",
            "billed_tokens",
            "dispatch_time",
            "first_token_time",
            "end_time",
            "ttft_s",
            "e2e_latency_s",
            "tpot_s",
            "error",
        ])
        for r in results:
            writer.writerow([
                r.engine,
                r.rate,
                r.request_id,
                int(r.success),
                r.http_status,
                r.prompt_chars,
                r.input_tokens,
                r.output_tokens,
                r.billed_tokens,
                f"{r.dispatch_time:.9f}",
                "" if r.first_token_time is None else f"{r.first_token_time:.9f}",
                f"{r.end_time:.9f}",
                "" if r.ttft_s is None else f"{r.ttft_s:.9f}",
                f"{r.e2e_latency_s:.9f}",
                "" if r.tpot_s is None else f"{r.tpot_s:.9f}",
                r.error,
            ])


def write_csv_aggregates(path: Path, aggregates: List[AggregateMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(aggregates[0]).keys()) if aggregates else [])
        if aggregates:
            writer.writeheader()
            for row in aggregates:
                writer.writerow(asdict(row))


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def plot_frontier(path: Path, aggregates: List[AggregateMetrics], args: argparse.Namespace) -> None:
    plt.figure(figsize=(10, 6))

    engines = sorted({a.engine for a in aggregates})
    markers = ["o", "s", "^", "D", "P", "X"]
    marker_map = {engine: markers[i % len(markers)] for i, engine in enumerate(engines)}

    for engine in engines:
        rows = sorted([a for a in aggregates if a.engine == engine], key=lambda a: a.rate)
        xs = [a.p95_ttft_ms for a in rows if a.p95_ttft_ms is not None]
        ys = [a.output_goodput_toks_per_s for a in rows if a.p95_ttft_ms is not None]
        labels = [a.rate for a in rows if a.p95_ttft_ms is not None]
        if not xs:
            continue
        plt.plot(xs, ys, marker=marker_map[engine], linewidth=2, label=engine)
        for x, y, r in zip(xs, ys, labels):
            plt.annotate(f"{r:g} rps", (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)

    plt.xlabel("p95 TTFT (ms) — lower is better")
    plt.ylabel("Successful output-token goodput (tokens/s) — higher is better")
    plt.title("Inference efficiency frontier: better systems move up and left")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def summarize_winner(aggregates: List[AggregateMetrics], friendli_label: str, oss_label: str) -> Dict[str, Any]:
    by_rate_engine: Dict[Tuple[float, str], AggregateMetrics] = {(a.rate, a.engine): a for a in aggregates}
    shared_rates = sorted({a.rate for a in aggregates if (a.rate, friendli_label) in by_rate_engine and (a.rate, oss_label) in by_rate_engine})
    comparisons = []
    for rate in shared_rates:
        f = by_rate_engine[(rate, friendli_label)]
        o = by_rate_engine[(rate, oss_label)]
        comparisons.append({
            "rate": rate,
            "friendli_p95_ttft_ms": f.p95_ttft_ms,
            "oss_p95_ttft_ms": o.p95_ttft_ms,
            "friendli_goodput_toks_per_s": f.output_goodput_toks_per_s,
            "oss_goodput_toks_per_s": o.output_goodput_toks_per_s,
            "friendli_billed_toks_per_resolved_turn": f.billed_toks_per_resolved_turn,
            "oss_billed_toks_per_resolved_turn": o.billed_toks_per_resolved_turn,
            "goodput_ratio_friendli_over_oss": (f.output_goodput_toks_per_s / o.output_goodput_toks_per_s) if o.output_goodput_toks_per_s > 0 else None,
            "p95_ttft_ratio_oss_over_friendli": (o.p95_ttft_ms / f.p95_ttft_ms) if (f.p95_ttft_ms and f.p95_ttft_ms > 0 and o.p95_ttft_ms) else None,
        })
    return {"comparisons": comparisons}


async def async_main(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    oss = EngineConfig(args.oss_label, args.oss_base_url, args.oss_model, args.oss_api_key)
    friendli = EngineConfig(args.friendli_label, args.friendli_base_url, args.friendli_model, args.friendli_api_key)
    rates = [float(x) for x in args.request_rates.split(",") if x.strip()]

    total_prompts_needed = max(args.requests_per_rate, args.warmup_requests) * len(rates)
    prompts = load_prompts(args.prompt_file, total_prompts_needed, args.seed)
    token_counter = TokenCounter(args.tokenizer)

    print(f"Loaded {len(prompts)} prompts")
    print(f"Benchmarking {oss.label}...")
    oss_results = await run_rate_sweep_for_engine(oss, token_counter, prompts, rates, args)
    print(f"Benchmarking {friendli.label}...")
    friendli_results = await run_rate_sweep_for_engine(friendli, token_counter, prompts, rates, args)

    all_results = oss_results + friendli_results
    aggregates = aggregate_results(all_results)

    write_csv_results(outdir / "request_results.csv", all_results)
    write_csv_aggregates(outdir / "aggregate_metrics.csv", aggregates)
    plot_frontier(outdir / "latency_throughput_frontier.png", aggregates, args)
    summary = {
        "args": vars(args),
        "winner_summary": summarize_winner(aggregates, args.friendli_label, args.oss_label),
    }
    write_json(outdir / "summary.json", summary)

    print(f"Wrote: {outdir / 'request_results.csv'}")
    print(f"Wrote: {outdir / 'aggregate_metrics.csv'}")
    print(f"Wrote: {outdir / 'latency_throughput_frontier.png'}")
    print(f"Wrote: {outdir / 'summary.json'}")


def main() -> None:
    args = parse_args()
    trio.run(async_main, args)


if __name__ == "__main__":
    main()
