# safe-llm

A layered safety harness around any local LLM. Three checkpoints —
**pre-generation**, **during-generation**, **post-generation** — guard every
turn, with a full audit trace per request.

Derived from Track I (Safety Harness) of the
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum (lessons 82–87), then hardened with Detoxify, Presidio,
sentence-transformers, and a DeBERTa injection detector.

## Requirements

- Python 3.10+
- [Ollama](https://ollama.com/) running locally with at least one model pulled — the default backend
- [uv](https://docs.astral.sh/uv/) (recommended) or plain `pip`

```bash
ollama pull llama3.2:3b
```

## The three checkpoints

| Stage | Runs on | Checks | Action on hit |
|---|---|---|---|
| **Pre-gen** (`safe_llm.detector`) | user prompt | substring + regex rules and (optionally) `deepset/deberta-v3-base-injection` | block before the model runs |
| **During-gen** (`safe_llm.gate`) | streamed tokens | Detoxify over the trailing char window every N chunks | terminate the stream mid-flight |
| **Post-gen** (`safe_llm.classifiers` + `safe_llm.rules`) | full completion | Detoxify (toxicity), Presidio (PII), sentence-transformers (system-prompt leakage) → YAML constitution rules | allow / redact / block / rewrite |

Every request writes a `SafetyTrace` to `outputs/<timestamp>/gate_trace.json`.

## Install

```bash
cd projects/safe-llm
uv sync --extra all         # everything: classifiers, PII, eval, CLI, server
# or à la carte:
uv sync --extra classifiers --extra pii --extra eval
```

Extras: `classifiers` (Detoxify, sentence-transformers, transformers, torch,
nltk — ~1 GB of weights on first use), `pii` (Presidio + spaCy), `eval`
(HuggingFace `datasets`), `cli`, `server`, `yaml`, `all`.

Plain pip works too:

```bash
pip install -e '.[all]'
```

## Commands

`safe-llm` is a console script registered by the package entry point — after
`uv sync` (or `pip install`) it is on your `PATH`. Equivalent invocations:
`safe-llm ...`, `uv run safe-llm ...`, `python -m safe_llm ...`.

### `chat` — interactive REPL through the gate

```bash
safe-llm chat --model llama3.2:3b
safe-llm chat --model llama3.2:3b --constitution examples/constitution.yml
```

Streams the model's response through all three checkpoints and writes a
`gate_trace.json` on exit.

### `eval` — batch evaluation on attack corpora

Single entry point for evaluation. Runs the gate against labeled fixtures,
writes the full trace, and prints an over/under-refusal summary.

```bash
# Default: mix all three corpora, sample 10
safe-llm eval --model llama3.2:3b

# Explicit corpora, larger sample, ML injection detector on
safe-llm eval --model llama3.2:3b --limit 50 \
    --sources hand-crafted jailbreakbench xstest \
    --use-ml-detector

# Bring your own fixtures (supersedes --sources)
safe-llm eval --model llama3.2:3b --fixtures my_prompts.json
```

Corpora:

| Source | Size | Fetch |
|---|---|---|
| `hand-crafted` | 50 | bundled (offline) |
| `jailbreakbench` | ~100 | HuggingFace `datasets`, cached |
| `xstest` | ~250 safe prompts | GitHub CSV (no HF auth) |

Each item is tagged `label="safe"` or `label="unsafe"`, so the summary
cleanly separates over-refusal (safe prompts blocked) from under-refusal
(unsafe prompts allowed), broken down by source.

### Post-hoc trace analysis

`safe-llm eval` already prints the analyzer summary. To analyze an existing
trace file:

```python
from pathlib import Path
from safe_llm.eval import analyze_traces, load_trace_file

traces = load_trace_file(Path("outputs/26-07-06_144854/gate_trace.json"))
report = analyze_traces(traces)
print(report.over_refusal_rate, report.under_refusal_rate, report.by_source)
```

## Example constitution

An example constitution lives at [`examples/constitution.yml`](examples/constitution.yml).
It is the same rule set the engine loads by default and is a good starting
template for your own — pass it to either subcommand:

```bash
safe-llm chat --model llama3.2:3b --constitution examples/constitution.yml
safe-llm eval --model llama3.2:3b --constitution examples/constitution.yml
```

Copy it, edit the rules, keep the schema (`name`, `severity`, `must`,
`explanation`, optional `applies_when` and `fix`) — the engine validates on
load and rejects malformed rules.

## HTTP server (OpenAI-compatible)

Any client that speaks OpenAI's Chat Completions API can point at the
harness:

```bash
SAFE_LLM_BACKEND=ollama SAFE_LLM_MODEL=llama3.2:3b \
    uvicorn safe_llm.server:app --host 0.0.0.0 --port 8765
```

| Env var | Default | Meaning |
|---|---|---|
| `SAFE_LLM_BACKEND` | `ollama` | `ollama` or `mock` |
| `SAFE_LLM_MODEL` | `llama3.2:3b` | model name passed to the backend |
| `SAFE_LLM_OLLAMA_HOST` | `http://localhost:11434` | Ollama HTTP host |

Endpoints:

- `GET /health` — returns backend/model info.
- `POST /v1/chat/completions` — OpenAI-shaped request/response. Streaming
  (`stream: true`) returns SSE. Every response includes the full audit
  trace as `x_safety_trace`.

The ML injection detector is off in server mode (avoids a ~500 MB model
download at startup); use `safe-llm eval --use-ml-detector` to benchmark it.

## Optional ML injection detector

`deepset/deberta-v3-base-injection` can be layered onto the pre-gen
`Detector` alongside the substring/regex rules — the verdict takes the max
confidence across the two signals. Requires the `classifiers` extra;
silently no-ops if `transformers` is not installed.

```python
from safe_llm import SafeLLM, SafetyGate
safe = SafeLLM(backend, gate=SafetyGate(use_ml_detector=True))
```

```bash
safe-llm chat --model llama3.2:3b --use-ml-detector
safe-llm eval --model llama3.2:3b --use-ml-detector
```

## Tests

The test suite is the fastest way to check that your install works, and
doubles as small runnable examples of each module. Every file under
`tests/` pins the contract of one piece of the harness:

| Test file | What it guards |
|---|---|
| `test_taxonomy.py` | The 6-category / 50-fixture jailbreak corpus and matcher |
| `test_detector.py` | Pre-gen prompt-injection detector — rules + ML scorer wiring |
| `test_classifiers.py` | Post-gen Detoxify / Presidio / leakage classifiers + severity router |
| `test_rules.py` | YAML constitution loader, predicate engine, fixer, diff |
| `test_refusal.py` | Refusal classifier, ECE calibration, mock policies |
| `test_gate.py` | End-to-end three-checkpoint gate + streaming termination |
| `test_gate_use_ml_detector.py` | The `use_ml_detector` flag / `--use-ml-detector` CLI switch |
| `test_eval_datasets.py` | Corpus loaders (hand-crafted, JailbreakBench, XSTest) |
| `test_eval_report.py` | Trace-file analyzer (over/under-refusal rollups) |
| `test_mock_backend.py` | `MockBackend` — lets you exercise the gate with no Ollama running |

The suite is split into two tiers, gated by an env var:

**Fast suite** — the default. Runs fully offline, no model downloads, no
network, ~121 tests in under a second. Use this as a sanity check after
installing to confirm the entry point resolves, the classifiers wire up,
and the gate composes correctly:

```bash
python -m unittest discover -s tests -t .
```

**Heavy suite** — additionally loads the real Detoxify / Presidio /
sentence-transformers / DeBERTa models and hits the HuggingFace Hub for
JailbreakBench / XSTest. This is where you find out whether your extras
resolved end-to-end, not just at import time. Expect a ~1 GB download on
first run. Set the env var *and* install the corresponding extras:

```bash
SAFE_LLM_HEAVY_TESTS=1 python -m unittest discover -s tests -t .
```

Heavy tests are individually skipped when the required import is missing,
so `--extra classifiers` alone (or any partial combination) still runs
cleanly — you only get the tests you have the models for.

If you plan to customize the harness — writing your own constitution,
swapping in a different injection detector, adding a backend — the
matching test file is the shortest path to seeing how that module is
called and what invariants it expects.

## License

MIT. Harness code is derived from the
[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
curriculum (Copyright © 2026 Rohit Ghumare).
