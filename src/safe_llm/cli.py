from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from .eval.datasets import EvalItem, KNOWN_SOURCES, load_combined
from .eval.report import analyze_traces
from .gate import SafetyGate, trace_to_dict
from .llm.ollama import OllamaBackend
from .safe_llm import SafeLLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"


def _build_safellm(model_name: str, use_ml_detector: bool) -> SafeLLM:
    backend = OllamaBackend(model_name)
    return SafeLLM(backend, gate=SafetyGate(use_ml_detector=use_ml_detector))


def chat_mode(model_name: str, output_dir=None, constitution_path=None, use_ml_detector: bool = False):
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR) / datetime.now().strftime("%y-%m-%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    safellm = _build_safellm(model_name, use_ml_detector)
    if constitution_path is not None:
        safellm.upload_constitution(Path(constitution_path))

    messages: list[dict] = []
    traces = []

    while True:
        user_input = input("\nUSER:")
        if "exit" in user_input.lower():
            break
        messages.append({"role": "user", "content": user_input})
        for chunk in safellm.stream_chat(messages):
            sys.stdout.write(chunk)
            sys.stdout.flush()
        print()

        if safellm.last_trace is not None:
            traces.append(safellm.last_trace)

    trace_path = output_dir / "gate_trace.json"
    payload = {
        "model": model_name,
        "turns": len(traces),
        "traces": [trace_to_dict(t) for t in traces],
    }
    trace_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nWrote {len(traces)} trace(s) to {trace_path}")


def _load_from_json(fixtures_path: str) -> list[EvalItem]:
    """Bring-your-own JSON fixtures.

    Accepts either the internal Fixture schema (id/category/subtype/prompt/
    target_behavior/severity, all treated as unsafe) or a richer EvalItem
    schema (id/source/prompt/category/label/severity).
    """
    with open(fixtures_path, "r") as f:
        raw = json.load(f)
    items: list[EvalItem] = []
    for i, d in enumerate(raw):
        items.append(
            EvalItem(
                id=str(d.get("id", f"custom-{i:03d}")),
                source=str(d.get("source", "custom")),
                prompt=str(d["prompt"]),
                category=str(d.get("category", "unknown")),
                label=str(d.get("label", "unsafe")),
                severity=d.get("severity"),
            )
        )
    return items


def _print_report_summary(traces: list[dict]) -> None:
    report = analyze_traces(traces)
    if report.n_safe == 0 and report.n_unsafe == 0:
        return
    print("─" * 80)
    print("Summary")
    print(f"  n_safe             = {report.n_safe}")
    print(f"  n_unsafe           = {report.n_unsafe}")
    print(f"  over_refusal_rate  = {report.over_refusal_rate:.3f}  (safe items blocked)")
    print(f"  under_refusal_rate = {report.under_refusal_rate:.3f}  (unsafe items allowed)")
    if report.by_source:
        print("\n  by source:")
        print(f"    {'source':16} {'n_safe':>7} {'n_unsafe':>9} {'over':>7} {'under':>7}")
        for src, stats in sorted(report.by_source.items()):
            print(
                f"    {src:16} {stats['n_safe']:>7} {stats['n_unsafe']:>9} "
                f"{stats['over_refusal_rate']:>7.3f} {stats['under_refusal_rate']:>7.3f}"
            )


def eval_mode(
    model_name: str,
    output_dir=None,
    fixtures_path=None,
    limit: int = 10,
    constitution_path=None,
    sources: list[str] | None = None,
    use_ml_detector: bool = False,
):
    if output_dir is None:
        output_dir = Path(OUTPUT_DIR) / datetime.now().strftime("%y-%m-%d_%H%M%S")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    if fixtures_path is not None:
        eval_items = _load_from_json(fixtures_path)
        active_sources = ["custom"]
    else:
        active_sources = list(sources) if sources else list(KNOWN_SOURCES)
        eval_items = load_combined(active_sources)

    if not eval_items:
        print("No eval items loaded — nothing to run.", file=sys.stderr)
        return

    if limit and len(eval_items) > limit:
        eval_items = rng.sample(eval_items, limit)
    rng.shuffle(eval_items)

    safellm = _build_safellm(model_name, use_ml_detector)
    if constitution_path is not None:
        safellm.upload_constitution(Path(constitution_path))

    SYSTEM_PROMPT = "Be concise. Keep every response within 3 to 5 sentences."

    print("SafeLLM Eval")
    print(f"  backend       : {safellm.backend.model}")
    print(f"  ml_detector   : {'on' if use_ml_detector else 'off'}")
    print(f"  sources       : {', '.join(active_sources)}")
    print(f"  prompts       : {len(eval_items)}")
    print()

    per_source_counts = {s: 0 for s in active_sources}
    for item in eval_items:
        per_source_counts[item.source] = per_source_counts.get(item.source, 0) + 1
    print("  breakdown by source:")
    for s, n in per_source_counts.items():
        print(f"    {s:16} {n}")
    print()

    traces: list[dict] = []
    for i, item in enumerate(eval_items, 1):
        sev_str = str(item.severity) if item.severity is not None else "—"
        prompt_preview = item.prompt if len(item.prompt) <= 90 else item.prompt[:87] + "..."

        print("─" * 80)
        print(f"  [{i:02d}] source      : {item.source}")
        print(f"       category    : {item.category}")
        print(f"       expected    : {item.label}")
        print(f"       severity    : {sev_str}")
        print(f"       prompt      : {prompt_preview}")
        print(f"       stream      : ", end="", flush=True)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item.prompt},
        ]
        chunk_count = 0
        for chunk in safellm.stream_chat(messages):
            sys.stdout.write(chunk)
            sys.stdout.flush()
            chunk_count += 1
        print()

        trace = safellm.last_trace
        assert trace is not None
        traces.append(
            {
                "item_id": item.id,
                "source": item.source,
                "category": item.category,
                "label": item.label,
                "severity": item.severity,
                **trace_to_dict(trace),
            }
        )
        print(f"       final_action: {trace.final_action}")
        print(f"       latency_ms  : {trace.latency_ms}")
        print(f"       streamed    : {chunk_count} chunks")
        print(f"       final_output: {trace.final_output}")
        print()

    trace_path = output_dir / "gate_trace.json"
    payload = {
        "model": model_name,
        "sources": active_sources,
        "turns": len(traces),
        "traces": traces,
    }
    trace_path.write_text(json.dumps(payload, indent=2) + "\n")

    _print_report_summary(traces)
    print(f"\nWrote {len(traces)} trace(s) to {trace_path}")


def cli():
    """Entrypoint for `safe-llm` (also available as `python -m safe_llm`).

    Subcommands:
        chat : interactive REPL against an Ollama-backed model, wrapped in
               the safety harness.
        eval : batch evaluation across hand-crafted, JailbreakBench, and
               XSTest corpora (mix and match via --sources). Also accepts a
               user-supplied JSON via --fixtures which supersedes --sources.
               Prints an over/under-refusal summary at the end.

    JSON fixtures schema (either shape is accepted):
        internal : id/category/subtype/prompt/target_behavior/severity
        eval     : id/source/prompt/category/label/severity
    """
    global_parser = argparse.ArgumentParser(description="Run a Safe LLM")
    sub = global_parser.add_subparsers(dest="cmd", required=True)

    ml_help = (
        "Load deepset/deberta-v3-base-injection into the pre-gen Detector "
        "alongside rules; take max confidence. Requires the [classifiers] "
        "extra."
    )

    chat_parser = sub.add_parser("chat", help="Interactive REPL")
    chat_parser.add_argument("--output-dir", type=str)
    chat_parser.add_argument("--model", type=str, required=True)
    chat_parser.add_argument("--constitution", type=str)
    chat_parser.add_argument("--use-ml-detector", action="store_true", help=ml_help)

    eval_parser = sub.add_parser("eval", help="Evaluate fixtures on a Safe LLM")
    eval_parser.add_argument("--model", type=str, required=True)
    eval_parser.add_argument("--output-dir", type=str)
    eval_parser.add_argument(
        "--fixtures",
        type=str,
        help="Path to a JSON file of custom eval items. Supersedes --sources.",
    )
    eval_parser.add_argument("--limit", type=int, default=10)
    eval_parser.add_argument("--constitution", type=str)
    eval_parser.add_argument(
        "--sources",
        nargs="+",
        choices=list(KNOWN_SOURCES),
        default=list(KNOWN_SOURCES),
        help=(
            "One or more datasets to sample from. Ignored when --fixtures is "
            "supplied. Defaults to all three."
        ),
    )
    eval_parser.add_argument("--use-ml-detector", action="store_true", help=ml_help)

    args = global_parser.parse_args()

    if args.cmd == "chat":
        chat_mode(
            args.model,
            args.output_dir,
            args.constitution,
            use_ml_detector=args.use_ml_detector,
        )
    elif args.cmd == "eval":
        eval_mode(
            model_name=args.model,
            output_dir=args.output_dir,
            fixtures_path=args.fixtures,
            limit=args.limit,
            constitution_path=args.constitution,
            use_ml_detector=args.use_ml_detector,
            sources=args.sources,
        )


if __name__ == "__main__":
    cli()
