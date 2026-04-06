# Benchmark README

This README covers:
- local **vLLM** setup in Bash
- Python dependencies for the benchmark script
- every CLI parameter supported by `benchmark_friendli_vs_open_source_v4.py`
- example benchmark commands
- output files and interpretation notes

## What this benchmark does

The script benchmarks two **OpenAI-compatible** inference endpoints under the same workload:
- an open-source engine such as **vLLM**
- **Friendli Engine**

It uses:
- **Trio** for orchestration and concurrency
- **aiohttp** for HTTP streaming
- **trio-asyncio** as the bridge layer because `aiohttp` is asyncio-native

The benchmark measures the most useful inference-efficiency metrics for agent-style workloads:
- p95 TTFT
- output-token goodput
- success rate
- billed tokens per resolved turn
- plus supporting latency and throughput metrics

It also generates a single chart:
- **latency_throughput_frontier.png**
- x-axis = **p95 TTFT (ms)**
- y-axis = **output-token goodput (tokens/sec)**
- better systems move **up and left**

---

## 1) Local vLLM setup (Bash)

### Option A: install vLLM with pip

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install vllm
```

### Option B: install the benchmark dependencies too

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install vllm trio trio-asyncio aiohttp numpy matplotlib transformers
```

### Start a local vLLM OpenAI-compatible server

Replace the model name with the exact model you want to serve.

```bash
source .venv/bin/activate
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

By default, vLLM serves locally on:

```bash
http://localhost:8000/v1
```

### Optional: choose host and port explicitly

```bash
source .venv/bin/activate
vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-abc123
```

### Quick health check with curl

```bash
curl http://localhost:8000/v1/models \
  -H 'Authorization: Bearer token-abc123'
```

---

## 2) Benchmark script dependencies

If you already have Python available, install the benchmark dependencies:

```bash
source .venv/bin/activate
pip install trio trio-asyncio aiohttp numpy matplotlib transformers
```

If you also want local vLLM in the same environment:

```bash
source .venv/bin/activate
pip install vllm trio trio-asyncio aiohttp numpy matplotlib transformers
```

---

## 3) Input prompt file format

The benchmark accepts a JSONL file where each line looks like this:

```json
{"prompt": "Find the bug in this Python function and propose a minimal patch."}
```

If `--prompt-file` is omitted, the script falls back to its built-in sample prompts.

---

## 4) Full benchmark command

```bash
python benchmark_friendli_vs_open_source_v4.py \
  --oss-label vLLM \
  --oss-base-url http://localhost:8000/v1 \
  --oss-model NousResearch/Meta-Llama-3-8B-Instruct \
  --oss-api-key token-abc123 \
  --friendli-label Friendli \
  --friendli-base-url https://inference.friendli.ai/v1 \
  --friendli-model meta-llama-3.1-8b-instruct \
  --friendli-api-key "$FRIENDLI_API_KEY" \
  --tokenizer NousResearch/Meta-Llama-3-8B-Instruct \
  --prompt-file sample_prompts_code.jsonl \
  --request-rates 1,2,4,8,16 \
  --requests-per-rate 50 \
  --warmup-requests 5 \
  --max-tokens 192 \
  --temperature 0.0 \
  --top-p 1.0 \
  --timeout-s 180 \
  --max-concurrency 256 \
  --seed 7 \
  --fixed-seed-param 7 \
  --system-prompt "You are a precise coding assistant. Return concise, factual answers." \
  --outdir ./bench_results
```

---

## 5) All execution parameters

Below is every CLI parameter supported by `benchmark_friendli_vs_open_source_v4.py`.

### Open-source endpoint parameters

#### `--oss-label`
Human-readable label for the open-source engine in CSVs and plots.

Default:
```bash
vLLM
```

#### `--oss-base-url`
Required. OpenAI-compatible base URL for the open-source engine, ending in `/v1`.

Example:
```bash
--oss-base-url http://localhost:8000/v1
```

#### `--oss-model`
Required. Model ID to send in the request payload for the open-source endpoint.

Example:
```bash
--oss-model NousResearch/Meta-Llama-3-8B-Instruct
```

#### `--oss-api-key`
API key for the open-source endpoint.

Default:
```bash
token-abc123
```

---

### Friendli endpoint parameters

#### `--friendli-label`
Human-readable label for Friendli results in CSVs and plots.

Default:
```bash
Friendli
```

#### `--friendli-base-url`
Required. Friendli OpenAI-compatible base URL, ending in `/v1`.

Example:
```bash
--friendli-base-url https://inference.friendli.ai/v1
```

#### `--friendli-model`
Required. Model ID to send in the request payload for Friendli.

Example:
```bash
--friendli-model meta-llama-3.1-8b-instruct
```

#### `--friendli-api-key`
Required. Friendli API key.

Example:
```bash
--friendli-api-key "$FRIENDLI_API_KEY"
```

---

### Shared workload parameters

#### `--tokenizer`
Required. Hugging Face tokenizer identifier or local tokenizer path used for token counting.

Example:
```bash
--tokenizer NousResearch/Meta-Llama-3-8B-Instruct
```

#### `--prompt-file`
Optional. Path to a JSONL prompt file with one object per line containing `{"prompt": ...}`.

Default:
```bash
None
```

#### `--request-rates`
Comma-separated offered request rates for the load sweep.

Default:
```bash
1,2,4,8,16
```

Example:
```bash
--request-rates 1,2,4,8,16,32
```

#### `--requests-per-rate`
Number of measured requests to send at each offered rate.

Default:
```bash
40
```

#### `--warmup-requests`
Number of warmup requests to send per engine before measured runs start.

Default:
```bash
3
```

#### `--max-tokens`
Maximum output tokens requested per completion.

Default:
```bash
192
```

#### `--temperature`
Sampling temperature.

Default:
```bash
0.0
```

#### `--top-p`
Top-p sampling parameter.

Default:
```bash
1.0
```

#### `--timeout-s`
Per-request timeout in seconds.

Default:
```bash
180.0
```

#### `--max-concurrency`
Upper bound on in-flight requests inside the benchmark client.

Default:
```bash
256
```

#### `--seed`
Seed used by the benchmark for deterministic prompt scheduling and related client-side randomness.

Default:
```bash
7
```

#### `--fixed-seed-param`
Seed value injected into request payloads when the target endpoint supports a request-level `seed` parameter.

Default:
```bash
7
```

#### `--disable-seed-param`
Flag. If set, the benchmark does **not** include the request-level seed parameter.

Usage:
```bash
--disable-seed-param
```

#### `--system-prompt`
System prompt added to each chat completion request.

Default:
```bash
You are a precise coding assistant. Return concise, factual answers.
```

#### `--skip-warmup`
Flag. If set, the benchmark skips warmup requests.

Usage:
```bash
--skip-warmup
```

#### `--outdir`
Output directory for CSVs, JSON summary, and the chart.

Default:
```bash
bench_results
```

---

## 6) Minimal run example

```bash
python benchmark_friendli_vs_open_source_v4.py \
  --oss-base-url http://localhost:8000/v1 \
  --oss-model NousResearch/Meta-Llama-3-8B-Instruct \
  --friendli-base-url https://inference.friendli.ai/v1 \
  --friendli-model meta-llama-3.1-8b-instruct \
  --friendli-api-key "$FRIENDLI_API_KEY" \
  --tokenizer NousResearch/Meta-Llama-3-8B-Instruct
```

---

## 7) Recommended fair-comparison settings

Keep these fixed across both endpoints:
- same model weights, if possible
- same tokenizer
- same prompt set
- same request-rate ladder
- same `temperature`
- same `top_p`
- same `max_tokens`
- same system prompt
- same seed behavior
- same network placement as much as possible

If you change the model family between engines, the result becomes **engine + model behavior**, not a clean engine-only comparison.

---

## 8) Output files

After a run, the script writes:

### `request_results.csv`
Per-request raw results.

### `aggregate_metrics.csv`
Per-engine, per-rate aggregate metrics.

### `summary.json`
Pairwise comparison summary across shared rates.

### `latency_throughput_frontier.png`
Single graph with:
- x-axis = p95 TTFT (ms)
- y-axis = successful output-token goodput (tokens/sec)

---

## 9) What the chart means

The frontier chart is intentionally simple:
- **left** = lower tail latency
- **up** = higher useful throughput

So the better engine sits more **up and left**.

That chart is useful because it shows both interactive responsiveness and useful serving capacity on one page.

---

## 10) Common pitfalls

### Comparing different models
If the Friendli side and vLLM side use different models, the result mixes model behavior with engine behavior.

### Using a tokenizer that does not match the served model
That will distort token-based metrics.

### Benchmarking across very different network paths
Remote Friendli versus local vLLM includes network effects. That is fine if you care about real deployment experience, but it is not a pure engine-only measurement.

### Letting sampling vary
If `temperature` or `top_p` differ, the benchmark is not fair.

---

## 11) Example two-terminal workflow

### Terminal 1: start local vLLM

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install vllm trio trio-asyncio aiohttp numpy matplotlib transformers

vllm serve NousResearch/Meta-Llama-3-8B-Instruct \
  --dtype auto \
  --api-key token-abc123
```

### Terminal 2: run the benchmark

```bash
source .venv/bin/activate
export FRIENDLI_API_KEY='YOUR_KEY_HERE'

python benchmark_friendli_vs_open_source_v4.py \
  --oss-label vLLM \
  --oss-base-url http://localhost:8000/v1 \
  --oss-model NousResearch/Meta-Llama-3-8B-Instruct \
  --oss-api-key token-abc123 \
  --friendli-label Friendli \
  --friendli-base-url https://inference.friendli.ai/v1 \
  --friendli-model meta-llama-3.1-8b-instruct \
  --friendli-api-key "$FRIENDLI_API_KEY" \
  --tokenizer NousResearch/Meta-Llama-3-8B-Instruct \
  --prompt-file sample_prompts_code.jsonl \
  --request-rates 1,2,4,8,16 \
  --requests-per-rate 50 \
  --max-tokens 192 \
  --warmup-requests 5 \
  --outdir ./bench_results
```
