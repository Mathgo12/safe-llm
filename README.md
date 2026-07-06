# safe-llm

A layered safety harness wrapped around any local LLM. Three checkpoints —
**pre-generation**, **during-generation**, **post-generation** — guard every
turn, with a full audit trace written per request.

Originally adapted from Track I (Safety Harness) of the AI Engineering from
Scratch curriculum, lessons 82–87, then hardened with production-grade
classifiers (Detoxify, Presidio, sentence-transformers, DeBERTa injection
detector) and a three-corpus evaluation loop.

## The three checkpoints

| Stage | Runs on | Checks | Action on hit |
|---|---|---|---|
| **Pre-gen** (`safe_llm.detector`) | user prompt | substring + regex rules and (optionally) `deepset/deberta-v3-base-injection`; takes max confidence | block before the model runs |
| **During-gen** (`safe_llm.gate`) | streamed tokens | Detoxify over the trailing char window, every N chunks (default 20) | terminate the stream mid-flight |
| **Post-gen** (`safe_llm.classifiers` + `safe_llm.rules`) | full completion | Detoxify (multi-label toxicity), Presidio (PII), sentence-transformers (system-prompt leakage) → YAML constitution rules engine | allow / redact / block / rewrite per policy |

Every request produces a `SafetyTrace` with per-stage verdicts, latency, and
the final action — dumped to `outputs/<timestamp>/gate_trace.json`.

## Package layout

| Module | What it does |
|---|---|
| `safe_llm.taxonomy` | 6-category attack taxonomy, 50 hand-crafted fixtures, trigram-cosine matcher |
| `safe_llm.detector` | Rules + ML prompt-injection detector, per-category metrics |
| `safe_llm.classifiers` | Detoxify toxicity, Presidio PII, sentence-transformers leakage, severity router |
| `safe_llm.rules` | YAML constitution, predicate engine, declarative fixer, structured diff |
| `safe_llm.refusal` | Refusal evaluation framework: over-refusal, under-refusal, ECE |
| `safe_llm.gate` | Three-checkpoint composition with streaming termination and audit trace |
| `safe_llm.eval` | Corpus loaders: hand-crafted, JailbreakBench, XSTest |
| `safe_llm.llm` | Backend adapters — `OllamaBackend`, `MockBackend`, base `Protocol` |
| `safe_llm.safe_llm` | `SafeLLM` wrapper that couples a backend to the gate |
| `safe_llm.cli` | `safe-llm chat` / `safe-llm eval` entrypoints |
| `safe_llm.server` | (WIP) OpenAI-compatible `/v1/chat/completions` proxy |

## Install

Using [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
cd projects/safe-llm
uv sync --extra all               # everything: classifiers, PII, eval, CLI, server
# or a la carte:
uv sync --extra classifiers --extra pii --extra eval
```

Extras:

| Extra | Adds |
|---|---|
| `classifiers` | Detoxify, sentence-transformers, transformers, torch, nltk (~1 GB of model weights on first use) |
| `pii` | Presidio analyzer/anonymizer + spaCy |
| `eval` | HuggingFace `datasets` for JailbreakBench |
| `cli` | `click` |
| `server` | FastAPI + Uvicorn |
| `yaml` | `pyyaml` for constitution files |
| `all` | union of the above |

Plain pip works too:

```bash
pip install -e '.[all]'
```

You'll also want [Ollama](https://ollama.com/) running locally with at least
one model pulled:

```bash
ollama pull llama3.2:3b
```

## Commands

`safe-llm` is a console script registered by the package's entry point — after
`uv sync` (or `pip install`) it is on your `PATH` inside the venv. Equivalent
invocations: `safe-llm ...`, `uv run safe-llm ...`, `python -m safe_llm ...`.

### `chat` — interactive REPL through the gate

```bash
safe-llm chat --model llama3.2:3b
safe-llm chat --model llama3.2:3b --constitution path/to/constitution.yaml
```

Streams the model's response through the three checkpoints and writes a
`gate_trace.json` on exit.

### `eval` — batch evaluation on attack corpora

```bash
# Default: mix all three corpora, sample 10
safe-llm eval --model llama3.2:3b

# Pick corpora explicitly, larger sample
safe-llm eval --model llama3.2:3b --limit 50 \
    --sources hand-crafted jailbreakbench xstest

# Bring your own fixtures (supersedes --sources)
safe-llm eval --model llama3.2:3b --fixtures my_prompts.json
```

Corpora:

| Source | Size | Axis | Fetch |
|---|---|---|---|
| `hand-crafted` | 50 | injection-technique | offline (bundled) |
| `jailbreakbench` | ~100 | harm-type | HuggingFace `datasets` on first use, cached |
| `xstest` | ~250 safe prompts | over-refusal probes | GitHub CSV (no HF auth needed) |

Each corpus tags every item with `label="safe"` or `label="unsafe"`, so the
trace file cleanly separates over-refusal (safe prompts wrongly blocked) from
under-refusal (unsafe prompts wrongly answered).

### Refusal evaluation

Standalone report against a policy:

```bash
python -m safe_llm.refusal.evaluation
```

CI mode — non-zero exit if thresholds are exceeded:

```bash
python -m safe_llm.refusal.evaluation --ci \
    --policy MockPolicyStrict \
    --max-over-refusal 0.05 --max-under-refusal 0.15 --max-ece 0.15
```

The GitHub Actions workflow in `.github/workflows/refusal-eval.yml` runs this
on every PR touching `src/safe_llm/**` or `tests/**` and uploads the report.

### Per-module demos

Each L82–87 module has a runnable `__main__`:

```bash
python -m safe_llm.taxonomy.taxonomy
python -m safe_llm.detector.detector
python -m safe_llm.refusal.evaluation
python -m safe_llm.classifiers.router
python -m safe_llm.rules.engine
python -m safe_llm.gate
```

## Tests

Fast suite runs offline with stubbed models:

```bash
python -m unittest discover -s . -p "test_*.py" -t .
```

Model-loading and network-dependent tests are gated behind
`SAFE_LLM_HEAVY_TESTS=1` **and** the required import being available:

```bash
SAFE_LLM_HEAVY_TESTS=1 python -m unittest discover -s . -p "test_*.py" -t .
```

## License

The harness code is derived from the AI Engineering from Scratch curriculum
(MIT, Copyright © 2026 Rohit Ghumare).
