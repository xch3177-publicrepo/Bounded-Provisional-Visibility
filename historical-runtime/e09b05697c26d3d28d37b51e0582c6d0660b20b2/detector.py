#!/usr/bin/env python3
"""D1, the frozen recipe-specific feasibility detector.

This module is intentionally independent of experiment construction and
evaluation.  It accepts only the safe feature-path inputs described by the W2D
preregistration amendments.  In particular, harness identifiers are not part
of :meth:`D1Detector.score` or :meth:`D1Detector.content_features`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import unicodedata
from typing import Any, Callable, Mapping, Sequence

import numpy as np


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
EXPECTED_DIM = 384
K_NEIGHBORS = 10
MIN_STD = 1e-12


class DetectorInputError(ValueError):
    """Raised when a supposedly frozen detector input violates the contract."""


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        raise DetectorInputError("normalized text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DetectorInputError(f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise DetectorInputError(f"{name} must be finite")
    return result


def _unit_vector(value: Any, expected_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape != (expected_dim,):
        raise DetectorInputError(
            f"{name} has shape {arr.shape}; expected ({expected_dim},)"
        )
    if not np.isfinite(arr).all():
        raise DetectorInputError(f"{name} contains a non-finite coordinate")
    norm = float(np.linalg.norm(arr))
    if not math.isfinite(norm) or norm <= 0:
        raise DetectorInputError(f"{name} has zero or non-finite norm")
    # The data builder already checks normalization.  Normalizing again here
    # prevents float32 round-off from affecting a cosine without accepting a
    # materially malformed vector.
    if not math.isclose(norm, 1.0, rel_tol=1e-4, abs_tol=1e-5):
        raise DetectorInputError(f"{name} is not normalized (norm={norm:.9g})")
    return arr / norm


def _unit_matrix(value: Any, expected_dim: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != expected_dim:
        raise DetectorInputError(
            f"{name} has shape {arr.shape}; expected (n, {expected_dim})"
        )
    if not np.isfinite(arr).all():
        raise DetectorInputError(f"{name} contains a non-finite coordinate")
    norms = np.linalg.norm(arr, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
        raise DetectorInputError(f"{name} contains a zero or non-finite row")
    if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
        raise DetectorInputError(f"{name} contains a non-normalized row")
    return arr / norms[:, None]


@dataclass(frozen=True)
class SourceEvidence:
    credential_valid: bool
    provenance_consistent: bool
    embedding_model_revision: str

    @classmethod
    def from_value(cls, value: Any) -> "SourceEvidence":
        def checked_bool(candidate: Any, name: str) -> bool:
            # bool("false") is True.  Coercion here would turn a malformed
            # negative source-control record into an affirmative credential or
            # provenance decision, contradicting Family S's exact conjunction.
            if not isinstance(candidate, bool):
                raise DetectorInputError(f"{name} must be a JSON boolean")
            return candidate

        def checked_revision(candidate: Any) -> str:
            if not isinstance(candidate, str) or not candidate:
                raise DetectorInputError(
                    "embedding_model_revision must be a non-empty string"
                )
            return candidate

        if isinstance(value, Mapping):
            required = (
                "credential_valid",
                "provenance_consistent",
                "embedding_model_revision",
            )
            missing = [key for key in required if key not in value]
            if missing:
                raise DetectorInputError(
                    f"source evidence is missing {', '.join(missing)}"
                )
            return cls(
                credential_valid=checked_bool(
                    value["credential_valid"], "credential_valid"
                ),
                provenance_consistent=checked_bool(
                    value["provenance_consistent"], "provenance_consistent"
                ),
                embedding_model_revision=checked_revision(
                    value["embedding_model_revision"]
                ),
            )
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) != 3:
                raise DetectorInputError(
                    "source-evidence tuple must contain exactly three values"
                )
            return cls(
                checked_bool(value[0], "credential_valid"),
                checked_bool(value[1], "provenance_consistent"),
                checked_revision(value[2]),
            )
        raise DetectorInputError("source evidence must be a mapping or tuple")


@dataclass(frozen=True)
class ZComponent:
    mean: float
    std: float

    @classmethod
    def from_value(cls, value: Mapping[str, Any], name: str) -> "ZComponent":
        mean = _finite_float(value.get("mean"), f"{name}.mean")
        std = _finite_float(value.get("std"), f"{name}.std")
        if std < MIN_STD:
            raise DetectorInputError(f"{name}.std is below {MIN_STD}")
        return cls(mean, std)

    def to_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class ZStatistics:
    rep: ZComponent
    knn: ZComponent

    @classmethod
    def from_value(cls, value: Any) -> "ZStatistics":
        if isinstance(value, ZStatistics):
            return value
        if not isinstance(value, Mapping):
            raise DetectorInputError("z statistics must be a mapping")
        if isinstance(value.get("rep"), Mapping) and isinstance(
            value.get("knn"), Mapping
        ):
            return cls(
                ZComponent.from_value(value["rep"], "rep"),
                ZComponent.from_value(value["knn"], "knn"),
            )
        # Tolerate a flat artifact representation while emitting only the
        # canonical nested form.
        return cls(
            ZComponent.from_value(
                {"mean": value.get("rep_mean"), "std": value.get("rep_std")},
                "rep",
            ),
            ZComponent.from_value(
                {"mean": value.get("knn_mean"), "std": value.get("knn_std")},
                "knn",
            ),
        )

    def to_dict(self) -> dict[str, dict[str, float]]:
        return {"rep": self.rep.to_dict(), "knn": self.knn.to_dict()}


@dataclass(frozen=True)
class ContentFeatures:
    rep: float
    knn: float

    def to_dict(self) -> dict[str, float]:
        return {"rep": self.rep, "knn": self.knn}


@dataclass(frozen=True)
class DetectorResult:
    rep: float
    knn: float
    c_score: float
    family_s_affirms: bool
    family_c_affirms: bool
    promote: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "rep": self.rep,
            "knn": self.knn,
            "c_score": self.c_score,
            "family_s_affirms": self.family_s_affirms,
            "family_c_affirms": self.family_c_affirms,
            "promote": self.promote,
            "decision": "PROMOTE" if self.promote else "QUARANTINE",
        }


Embedder = Callable[[Sequence[str]], Any]


class D1Detector:
    """Frozen D1 score with an injectable embedder for deterministic tests."""

    def __init__(
        self,
        *,
        embedder: Embedder | None = None,
        model_name: str = MODEL_NAME,
        pinned_model_revision: str = MODEL_REVISION,
        expected_dim: int = EXPECTED_DIM,
        k_neighbors: int = K_NEIGHBORS,
    ) -> None:
        if expected_dim <= 0:
            raise DetectorInputError("expected_dim must be positive")
        if k_neighbors <= 0:
            raise DetectorInputError("k_neighbors must be positive")
        self.model_name = model_name
        self.pinned_model_revision = pinned_model_revision
        self.expected_dim = expected_dim
        self.k_neighbors = k_neighbors
        self._external_embedder = embedder
        self._model = None

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        if self._external_embedder is not None:
            raw = self._external_embedder(tuple(texts))
        else:
            if self._model is None:
                # Kept lazy so the verifier and unit tests can import this
                # module without loading torch or downloading a model.
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    self.model_name,
                    revision=self.pinned_model_revision,
                    trust_remote_code=False,
                )
            raw = self._model.encode(
                list(texts),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        arr = np.asarray(raw, dtype=np.float64)
        if arr.shape != (len(texts), self.expected_dim):
            raise DetectorInputError(
                f"embedder returned shape {arr.shape}; expected "
                f"({len(texts)}, {self.expected_dim})"
            )
        if not np.isfinite(arr).all():
            raise DetectorInputError("embedder returned a non-finite coordinate")
        norms = np.linalg.norm(arr, axis=1)
        if np.any(~np.isfinite(norms)) or np.any(norms <= 0):
            raise DetectorInputError("embedder returned a zero or invalid vector")
        return arr / norms[:, None]

    @staticmethod
    def _halves(text: str) -> tuple[str, str]:
        normalized = normalize_text(text)
        words = normalized.split()
        cut = len(words) // 2
        if cut == 0 or cut == len(words):
            raise DetectorInputError("text must have two non-empty word halves")
        return " ".join(words[:cut]), " ".join(words[cut:])

    def content_features(
        self,
        normalized_text: str,
        full_embedding: Any,
        reference_embeddings: Any,
        reference_groups: Sequence[str],
        excluded_source_group: str,
    ) -> ContentFeatures:
        """Compute ``rep`` and ``knn`` without a threshold or decision."""
        first, second = self._halves(normalized_text)
        halves = self._embed((first, second))
        rep = float(np.dot(halves[0], halves[1]))

        full = _unit_vector(full_embedding, self.expected_dim, "full embedding")
        refs = _unit_matrix(
            reference_embeddings, self.expected_dim, "reference embeddings"
        )
        if len(reference_groups) != refs.shape[0]:
            raise DetectorInputError(
                "reference_groups length does not match reference embeddings"
            )
        keep = np.fromiter(
            (str(group) != str(excluded_source_group) for group in reference_groups),
            dtype=bool,
            count=len(reference_groups),
        )
        usable = refs[keep]
        if usable.shape[0] < self.k_neighbors:
            raise DetectorInputError(
                f"only {usable.shape[0]} eligible references; "
                f"need {self.k_neighbors}"
            )
        similarities = usable @ full
        top = np.sort(similarities)[-self.k_neighbors :]
        knn = 1.0 - float(np.mean(top))
        if not math.isfinite(rep) or not math.isfinite(knn):
            raise DetectorInputError("content feature is non-finite")
        return ContentFeatures(rep=rep, knn=knn)

    @staticmethod
    def content_score(
        features: ContentFeatures, z_statistics: Any
    ) -> float:
        z = ZStatistics.from_value(z_statistics)
        score = (
            (features.rep - z.rep.mean) / z.rep.std
            + (features.knn - z.knn.mean) / z.knn.std
        )
        if not math.isfinite(score):
            raise DetectorInputError("content score is non-finite")
        return float(score)

    def source_affirms(
        self, source_evidence: Any, model_revision: str
    ) -> bool:
        evidence = SourceEvidence.from_value(source_evidence)
        return bool(
            evidence.credential_valid
            and evidence.provenance_consistent
            and evidence.embedding_model_revision == self.pinned_model_revision
            and str(model_revision) == self.pinned_model_revision
        )

    def score(
        self,
        normalized_text: str,
        full_embedding: Any,
        source_evidence: Any,
        reference_embeddings: Any,
        reference_groups: Sequence[str],
        excluded_source_group: str,
        model_revision: str,
        z_statistics: Any,
        threshold: float,
    ) -> DetectorResult:
        """Return a decision; no harness identifier is accepted."""
        threshold = _finite_float(threshold, "threshold")
        features = self.content_features(
            normalized_text,
            full_embedding,
            reference_embeddings,
            reference_groups,
            excluded_source_group,
        )
        c_score = self.content_score(features, z_statistics)
        family_s = self.source_affirms(source_evidence, model_revision)
        family_c = c_score < threshold
        return DetectorResult(
            rep=features.rep,
            knn=features.knn,
            c_score=c_score,
            family_s_affirms=family_s,
            family_c_affirms=family_c,
            promote=family_s and family_c,
        )

    def reference_z_statistics(
        self,
        reference_texts: Sequence[str],
        reference_embeddings: Any,
        reference_groups: Sequence[str],
    ) -> ZStatistics:
        """Compute population z-statistics from the frozen reference index."""
        refs = _unit_matrix(
            reference_embeddings, self.expected_dim, "reference embeddings"
        )
        if len(reference_texts) != refs.shape[0] or len(reference_groups) != refs.shape[0]:
            raise DetectorInputError(
                "reference texts, embeddings and groups must have equal length"
            )
        half_texts: list[str] = []
        for text in reference_texts:
            half_texts.extend(self._halves(text))
        half_embeddings = self._embed(half_texts).reshape(
            refs.shape[0], 2, self.expected_dim
        )
        rep = np.einsum(
            "ij,ij->i", half_embeddings[:, 0, :], half_embeddings[:, 1, :]
        )
        similarities = refs @ refs.T
        knn_values: list[float] = []
        group_array = np.asarray([str(group) for group in reference_groups])
        for index, group in enumerate(group_array):
            eligible = similarities[index, group_array != group]
            if eligible.size < self.k_neighbors:
                raise DetectorInputError(
                    f"reference row {index} has only {eligible.size} eligible "
                    f"neighbors; need {self.k_neighbors}"
                )
            top = np.partition(eligible, eligible.size - self.k_neighbors)[
                -self.k_neighbors :
            ]
            knn_values.append(1.0 - float(np.mean(top)))
        knn = np.asarray(knn_values, dtype=np.float64)
        rep_std = float(np.std(rep, ddof=0))
        knn_std = float(np.std(knn, ddof=0))
        if rep_std < MIN_STD or knn_std < MIN_STD:
            raise DetectorInputError(
                f"reference std below {MIN_STD}: rep={rep_std}, knn={knn_std}"
            )
        return ZStatistics(
            rep=ZComponent(float(np.mean(rep)), rep_std),
            knn=ZComponent(float(np.mean(knn)), knn_std),
        )
