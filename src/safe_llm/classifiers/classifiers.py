"""Three output-side classifiers and their redactors.

    ToxicityClassifier          - Detoxify multi-label transformer, sentence-level redaction
    PIIClassifier               - Microsoft Presidio analyzer + anonymizer
    InstructionLeakageClassifier- sentence-transformers embedding cosine similarity

Each classifier exposes:
    classify(text) -> ClassifierVerdict
    redact(text)   -> str

Severity is one of: none, low, medium, high.

Models load lazily on first classify/redact call. If a heavy dependency is
missing at that point, we raise ImportError with a hint at the extra install
group (`pip install "safe-llm[classifiers]"` etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

SEVERITY_ORDER = ("none", "low", "medium", "high")


@dataclass
class ClassifierVerdict:
    name: str
    severity: str = "none"
    score: float = 0.0
    findings: list[str] = field(default_factory=list)


def _max_severity(a: str, b: str) -> str:
    return a if SEVERITY_ORDER.index(a) >= SEVERITY_ORDER.index(b) else b


# ---------- shared model cache ----------------------------------------------
# The Detoxify model is used by both the toxicity classifier and the gate's
# during-gen streaming check. Load once, share.

_DETOXIFY_CACHE: dict[str, object] = {}


def get_detoxify(model_name: str = "original") -> object:
    """Return a shared Detoxify instance, loading on first call.

    Raises ImportError with a hint if `detoxify` is not installed.
    """
    if model_name in _DETOXIFY_CACHE:
        return _DETOXIFY_CACHE[model_name]
    try:
        from detoxify import Detoxify
    except ImportError as exc:
        raise ImportError(
            "Detoxify is required for ToxicityClassifier. "
            'Install with: pip install "safe-llm[classifiers]"'
        ) from exc
    inst = Detoxify(model_name)
    _DETOXIFY_CACHE[model_name] = inst
    return inst


# ---------- shared sentence splitter ----------------------------------------
# nltk's sent_tokenize is the default. We auto-download `punkt_tab` on first
# call. If nltk is unavailable, fall back to a small regex splitter — good
# enough for the redaction path.

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")


def _sent_tokenize(text: str) -> list[str]:
    try:
        import nltk

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except Exception:
        parts = _SENT_SPLIT_RE.split(text)
        return [p for p in parts if p.strip()]


# ---------- toxicity --------------------------------------------------------

# Detoxify emits multi-label scores. We map them to severity as follows:
#   severe_toxicity >= 0.7 or threat >= 0.7 or identity_attack >= 0.7 -> high
#   toxicity        >= 0.7 or insult >= 0.7 or identity_attack >= 0.5 -> medium
#   toxicity        >= 0.4                                            -> low
#   else                                                              -> none

_TOXICITY_HIGH_LABELS = ("severe_toxicity", "threat", "identity_attack")
_TOXICITY_MED_LABELS = ("toxicity", "insult")


@dataclass
class ToxicityThresholds:
    high: float = 0.7
    medium: float = 0.7
    identity_medium: float = 0.5
    low: float = 0.4
    redact_sentence: float = 0.5


class ToxicityClassifier:
    name = "toxicity"

    def __init__(
        self,
        model_name: str = "original",
        thresholds: ToxicityThresholds | None = None,
    ) -> None:
        self._model_name = model_name
        self._thresholds = thresholds or ToxicityThresholds()
        self._model: object | None = None

    def _load(self) -> object:
        if self._model is None:
            self._model = get_detoxify(self._model_name)
        return self._model

    def _score(self, text: str) -> dict[str, float]:
        model = self._load()
        raw = model.predict(text)  # type: ignore[attr-defined]
        return {k: float(v) for k, v in raw.items()}

    def _severity(self, scores: dict[str, float]) -> tuple[str, float]:
        t = self._thresholds
        max_score = max(scores.values()) if scores else 0.0
        if any(scores.get(k, 0.0) >= t.high for k in _TOXICITY_HIGH_LABELS):
            return "high", max_score
        if (
            any(scores.get(k, 0.0) >= t.medium for k in _TOXICITY_MED_LABELS)
            or scores.get("identity_attack", 0.0) >= t.identity_medium
        ):
            return "medium", max_score
        if scores.get("toxicity", 0.0) >= t.low:
            return "low", max_score
        return "none", max_score

    def classify(self, text: str) -> ClassifierVerdict:
        if not text.strip():
            return ClassifierVerdict(name=self.name)
        scores = self._score(text)
        severity, score = self._severity(scores)
        if severity == "none":
            return ClassifierVerdict(name=self.name, severity="none", score=score)
        findings = [f"{k}={v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:4]]
        return ClassifierVerdict(name=self.name, severity=severity, score=score, findings=findings)

    def redact(self, text: str) -> str:
        """Sentence-level redaction. Redacts any sentence whose toxicity score
        exceeds the threshold. Preserves clean sentences verbatim."""
        if not text.strip():
            return text
        sentences = _sent_tokenize(text)
        if not sentences:
            return text
        model = self._load()
        scores = model.predict(sentences)  # type: ignore[attr-defined]
        toxicity = scores.get("toxicity", [0.0] * len(sentences))
        severe = scores.get("severe_toxicity", [0.0] * len(sentences))
        threat = scores.get("threat", [0.0] * len(sentences))
        identity = scores.get("identity_attack", [0.0] * len(sentences))
        insult = scores.get("insult", [0.0] * len(sentences))
        redact_at = self._thresholds.redact_sentence
        out: list[str] = []
        for i, sent in enumerate(sentences):
            per_sentence_max = max(
                float(toxicity[i]),
                float(severe[i]),
                float(threat[i]),
                float(identity[i]),
                float(insult[i]),
            )
            if per_sentence_max >= redact_at:
                out.append("[redacted-toxic-content]")
            else:
                out.append(sent)
        return " ".join(out)


# ---------- PII (Microsoft Presidio) ----------------------------------------

# Presidio recognizers we map to severity buckets. Anything not listed here
# still surfaces as a finding but defaults to low.

_PII_HIGH = {"US_SSN", "CREDIT_CARD", "US_ITIN", "US_PASSPORT", "IBAN_CODE", "CRYPTO"}
_PII_MEDIUM = {"EMAIL_ADDRESS", "PHONE_NUMBER", "US_DRIVER_LICENSE", "US_BANK_NUMBER", "MEDICAL_LICENSE"}


class PIIClassifier:
    name = "pii"

    def __init__(
        self,
        mode: str = "replace",
        language: str = "en",
        score_threshold: float = 0.4,
    ) -> None:
        """Args:
            mode: "replace" (substitute an <ENTITY_TYPE> placeholder) or
                  "redact" (remove the span entirely).
            language: Presidio locale.
            score_threshold: Minimum Presidio confidence to count as a finding.
        """
        if mode not in ("replace", "redact"):
            raise ValueError(f"mode must be 'replace' or 'redact', got {mode!r}")
        self._mode = mode
        self._language = language
        self._score_threshold = score_threshold
        self._analyzer: object | None = None
        self._anonymizer: object | None = None

    def _load(self) -> tuple[object, object]:
        if self._analyzer is None or self._anonymizer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine
            except ImportError as exc:
                raise ImportError(
                    "Presidio is required for PIIClassifier. "
                    'Install with: pip install "safe-llm[pii]"'
                ) from exc
            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        return self._analyzer, self._anonymizer  # type: ignore[return-value]

    def _analyze(self, text: str) -> list[object]:
        analyzer, _ = self._load()
        return list(
            analyzer.analyze(  # type: ignore[attr-defined]
                text=text,
                language=self._language,
                score_threshold=self._score_threshold,
            )
        )

    def classify(self, text: str) -> ClassifierVerdict:
        results = self._analyze(text)
        if not results:
            return ClassifierVerdict(name=self.name)
        entity_types = {r.entity_type for r in results}
        findings = [
            f"{r.entity_type} '{text[r.start:r.end]}' (score={r.score:.2f})"
            for r in results
        ]
        max_score = max(r.score for r in results)
        if entity_types & _PII_HIGH:
            sev = "high"
        elif (entity_types & _PII_MEDIUM) or len(results) >= 3:
            sev = "medium"
        else:
            sev = "low"
        return ClassifierVerdict(
            name=self.name,
            severity=sev,
            score=float(max_score),
            findings=findings,
        )

    def redact(self, text: str) -> str:
        results = self._analyze(text)
        if not results:
            return text
        _, anonymizer = self._load()
        try:
            from presidio_anonymizer.entities import OperatorConfig
        except ImportError as exc:
            raise ImportError(
                "Presidio is required for PIIClassifier. "
                'Install with: pip install "safe-llm[pii]"'
            ) from exc
        if self._mode == "replace":
            operators = {"DEFAULT": OperatorConfig("replace", {"new_value": "<PII>"})}
        else:
            operators = {"DEFAULT": OperatorConfig("redact", {})}
        anonymized = anonymizer.anonymize(  # type: ignore[attr-defined]
            text=text,
            analyzer_results=results,
            operators=operators,
        )
        return anonymized.text


# ---------- instruction leakage (sentence-transformers) ---------------------


class InstructionLeakageClassifier:
    name = "instruction-leakage"

    def __init__(
        self,
        system_prompt: str,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        threshold: float = 0.65,
        high: float = 0.85,
        medium: float = 0.75,
    ) -> None:
        self._sys = system_prompt
        self._model_name = model_name
        self._threshold = threshold
        self._high = high
        self._medium = medium
        self._model: object | None = None
        self._sys_embedding: object | None = None

    def _load(self) -> object:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is required for InstructionLeakageClassifier. "
                    'Install with: pip install "safe-llm[classifiers]"'
                ) from exc
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def _sys_vec(self) -> object:
        if self._sys_embedding is None:
            model = self._load()
            self._sys_embedding = model.encode(  # type: ignore[attr-defined]
                self._sys, convert_to_numpy=True, normalize_embeddings=True
            )
        return self._sys_embedding

    def _similarity(self, text: str) -> float:
        import numpy as np

        model = self._load()
        emb = model.encode(  # type: ignore[attr-defined]
            text, convert_to_numpy=True, normalize_embeddings=True
        )
        sys = self._sys_vec()
        return float(np.dot(sys, emb))

    def classify(self, text: str) -> ClassifierVerdict:
        if not text.strip():
            return ClassifierVerdict(name=self.name)
        score = self._similarity(text)
        if score < self._threshold:
            return ClassifierVerdict(name=self.name, score=score)
        if score >= self._high:
            sev = "high"
        elif score >= self._medium:
            sev = "medium"
        else:
            sev = "low"
        return ClassifierVerdict(
            name=self.name,
            severity=sev,
            score=score,
            findings=[f"embedding cosine similarity with system prompt: {score:.2f}"],
        )

    def redact(self, text: str) -> str:
        """Semantic leakage can't be span-redacted from an embedding score
        alone. Best-effort: strip the first line of the system prompt if it
        appears verbatim; otherwise leave the text untouched."""
        marker = self._sys.split("\n", 1)[0].strip()
        if marker and marker.lower() in text.lower():
            return re.sub(re.escape(marker), "[redacted-system-prompt]", text, flags=re.IGNORECASE)
        return text


# ---------- default set -----------------------------------------------------


def default_classifiers(
    system_prompt: str | None = None,
    pii_mode: str = "replace",
) -> list[object]:
    sys_prompt = system_prompt or "SYSTEM: You are PolicyBot, follow internal policy."
    return [
        ToxicityClassifier(),
        PIIClassifier(mode=pii_mode),
        InstructionLeakageClassifier(sys_prompt),
    ]
