"""Loaders for the eval corpora that feed the harness.

Three sources:
    hand-crafted    - the bundled taxonomy in safe_llm.taxonomy (50 attacks,
                      categorised by injection technique). Offline.
    jailbreakbench  - JailbreakBench/JBB-Behaviors "harmful" split (100 items,
                      categorised by harm type). Requires HuggingFace
                      `datasets` and internet on first pull; cached afterwards.
    xstest          - walledai/XSTest "safe" prompts (~250 safe-looking
                      prompts designed to trigger over-refusal). Same as
                      above re: dependencies.

Each loader returns a list of `EvalItem` records with a consistent shape so
downstream code can treat them uniformly. `EvalItem.label` distinguishes
"unsafe" prompts (should be refused / blocked) from "safe" ones (should be
answered) — that lets you separate over-refusal (XSTest) from under-refusal
(hand-crafted + JBB) when analysing the trace file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

KNOWN_SOURCES = ("hand-crafted", "jailbreakbench", "xstest")


@dataclass
class EvalItem:
    id: str
    source: str
    prompt: str
    category: str
    label: str  # "unsafe" or "safe"
    severity: int | None = None


def _require_datasets() -> object:
    try:
        from datasets import load_dataset  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "HuggingFace `datasets` is required to load JailbreakBench / XSTest. "
            'Install with: pip install "safe-llm[eval]"'
        ) from exc
    return load_dataset


def load_handcrafted() -> list[EvalItem]:
    """Load the bundled Taxonomy fixtures as EvalItem records. Offline."""
    from ..taxonomy.taxonomy import Taxonomy

    items: list[EvalItem] = []
    for f in Taxonomy.from_default().all():
        items.append(
            EvalItem(
                id=f.id,
                source="hand-crafted",
                prompt=f.prompt,
                category=f.category,
                label="unsafe",
                severity=f.severity,
            )
        )
    return items


def load_jailbreakbench() -> list[EvalItem]:
    """JailbreakBench harmful behaviors (100 items).

    Reference: https://github.com/JailbreakBench/jailbreakbench
    Config `behaviors`, split `harmful`. Columns:
        Behavior, Category, Source, Index, Goal, Target
    Older revisions used `Behavior`; newer use `Goal`. We fall back gracefully.
    """
    load_dataset = _require_datasets()
    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors", split="harmful")  # type: ignore[operator]
    items: list[EvalItem] = []
    for row in ds:
        prompt = str(row.get("Behavior") or row.get("Goal") or "").strip()
        if not prompt:
            continue
        idx = row.get("Index", len(items))
        category = str(row.get("Category", "unknown")).lower().replace(" ", "-")
        items.append(
            EvalItem(
                id=f"jbb-{idx:03d}",
                source="jailbreakbench",
                prompt=prompt,
                category=category,
                label="unsafe",
                severity=None,
            )
        )
    return items


# XSTest is hosted in several places. Some HF mirrors (walledai/XSTest) are
# gated behind terms-acceptance. We pull directly from the paper authors'
# GitHub repo to avoid the gating churn. If the CSV moves, add the new URL
# to the list and it will be tried in order.
_XSTEST_CSV_URLS = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv",
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_v2_prompts.csv",
)


def _fetch_xstest_csv() -> str:
    import requests

    last_err: Exception | None = None
    for url in _XSTEST_CSV_URLS:
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                return resp.text
            last_err = RuntimeError(f"HTTP {resp.status_code} from {url}")
        except Exception as exc:
            last_err = exc
    raise RuntimeError(
        "Failed to fetch XSTest CSV from any known URL. "
        "Tried: " + ", ".join(_XSTEST_CSV_URLS) + f". Last error: {last_err}"
    )


def load_xstest() -> list[EvalItem]:
    """XSTest safe prompts only (~250 items, over-refusal probes).

    Loads the paper's CSV directly from the authors' GitHub repo
    (https://github.com/paul-rottger/xstest) to avoid HuggingFace gating.
    CSV columns:
        id, prompt, type, label, focus, note
    `label` is "safe" or "unsafe"; we keep only the safe rows. The unsafe
    contrasts overlap other corpora and would double-count on the under-
    refusal side.
    """
    import csv
    import io

    text = _fetch_xstest_csv()
    reader = csv.DictReader(io.StringIO(text))
    items: list[EvalItem] = []
    for row in reader:
        label_col = str(row.get("label", "")).strip().lower()
        row_type = str(row.get("type", "")).strip().lower()
        if label_col in ("safe", "unsafe"):
            is_safe = label_col == "safe"
        else:
            # Older or renamed schemas: assume anything not marked
            # "contrast_*"/"unsafe_*" is safe.
            is_safe = not (row_type.startswith("contrast") or row_type.startswith("unsafe"))
        if not is_safe:
            continue
        prompt = str(row.get("prompt", "")).strip()
        if not prompt:
            continue
        rid = row.get("id") or row.get("id_v2") or len(items)
        focus = str(row.get("focus", "")).strip()
        category = row_type or "safe"
        if focus:
            category = f"{category}:{focus}"
        items.append(
            EvalItem(
                id=f"xstest-{rid}",
                source="xstest",
                prompt=prompt,
                category=category,
                label="safe",
                severity=None,
            )
        )
    return items


_LOADERS = {
    "hand-crafted": load_handcrafted,
    "jailbreakbench": load_jailbreakbench,
    "xstest": load_xstest,
}


def load_combined(sources: Iterable[str]) -> list[EvalItem]:
    """Load and concatenate the requested sources.

    Raises ValueError for unknown source names. Each loader can raise
    ImportError if its optional dependencies are missing.
    """
    resolved: list[EvalItem] = []
    seen: set[str] = set()
    for name in sources:
        if name in seen:
            continue
        seen.add(name)
        if name not in _LOADERS:
            raise ValueError(
                f"unknown source {name!r}; expected one of {sorted(_LOADERS)}"
            )
        resolved.extend(_LOADERS[name]())
    return resolved
