"""End-to-end face recognition: detect -> align -> embed -> match.

A *gallery* maps person names to reference embeddings.  Matching is cosine
similarity; a face is labelled ``unknown`` when the best score falls below the
configured threshold (0.35 is a reasonable default for ArcFace-style models).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..models import find_face_model
from .align import (ARCFACE_TEMPLATE, FaceEmbedder, align_face,
                    embedding_similarity)
from .scrfd import Face, FaceDetector

__all__ = ["RecognizedFace", "FaceGallery", "FaceRecognizer", "FaceResult"]

DEFAULT_THRESHOLD = 0.35
UNKNOWN = "unknown"


@dataclass
class RecognizedFace:
    """A detected face plus its identity, if one was matched."""

    face: Face
    name: str = UNKNOWN
    similarity: float = 0.0
    embedding: Optional[np.ndarray] = None

    @property
    def box(self) -> np.ndarray:
        return self.face.box

    @property
    def score(self) -> float:
        return self.face.score

    def as_dict(self) -> dict:
        return {
            **self.face.as_dict(),
            "name": self.name,
            "similarity": round(float(self.similarity), 4),
        }


@dataclass
class FaceResult:
    """All faces found in one image."""

    faces: List[RecognizedFace]
    detector_backend: str = "numpy"
    embedder_backend: str = "numpy"
    inference_seconds: float = 0.0

    def __len__(self) -> int:
        return len(self.faces)

    def as_dict(self) -> dict:
        return {
            "count": len(self.faces),
            "backend": {"detector": self.detector_backend,
                        "embedder": self.embedder_backend},
            "inference_ms": round(self.inference_seconds * 1000, 2),
            "faces": [face.as_dict() for face in self.faces],
        }


class FaceGallery:
    """A set of named reference embeddings, persisted as ``.npz``."""

    def __init__(self, embeddings: Optional[np.ndarray] = None,
                 names: Optional[Sequence[str]] = None, threshold: float = DEFAULT_THRESHOLD):
        self.embeddings = (np.zeros((0, 512), dtype=np.float32)
                           if embeddings is None else np.asarray(embeddings, dtype=np.float32))
        self.names: List[str] = list(names or [])
        self.threshold = float(threshold)

    def __len__(self) -> int:
        return len(self.names)

    @property
    def people(self) -> List[str]:
        return sorted(set(self.names))

    # -- persistence -------------------------------------------------------- #
    @classmethod
    def load(cls, path: str, threshold: float = DEFAULT_THRESHOLD) -> "FaceGallery":
        if not os.path.exists(path):
            raise FileNotFoundError(f"gallery not found: {path}")
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            embeddings = np.asarray([entry["embedding"] for entry in payload["entries"]],
                                    dtype=np.float32)
            names = [entry["name"] for entry in payload["entries"]]
            return cls(embeddings, names, payload.get("threshold", threshold))

        with np.load(path, allow_pickle=False) as data:
            embeddings = np.asarray(data["embeddings"], dtype=np.float32)
            if "names_json" in data:
                # Names are stored as a JSON string: object arrays would require
                # pickle, which we refuse to load from disk.
                names = json.loads(str(data["names_json"].item()))
            else:
                names = [str(n) for n in data["names"]]
        return cls(embeddings, names, threshold)

    def save(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        if path.endswith(".json"):
            payload = {
                "threshold": self.threshold,
                "entries": [
                    {"name": name, "embedding": [float(v) for v in vector]}
                    for name, vector in zip(self.names, self.embeddings)
                ],
            }
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)
            return
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            names_json=np.asarray(json.dumps(self.names, ensure_ascii=False)),
            threshold=np.asarray(float(self.threshold)),
            format_version=np.asarray(1, dtype=np.int64),
        )

    # -- mutation ----------------------------------------------------------- #
    def add(self, name: str, embedding: np.ndarray) -> None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        if self.embeddings.size == 0:
            self.embeddings = vector
        elif self.embeddings.shape[1] != vector.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: gallery has "
                f"{self.embeddings.shape[1]}, got {vector.shape[1]}")
        else:
            self.embeddings = np.vstack([self.embeddings, vector])
        self.names.append(name)

    def remove(self, name: str) -> int:
        keep = [i for i, n in enumerate(self.names) if n != name]
        removed = len(self.names) - len(keep)
        self.names = [self.names[i] for i in keep]
        self.embeddings = (self.embeddings[keep] if keep
                           else np.zeros((0, self.embeddings.shape[1] if self.embeddings.size else 512),
                                         dtype=np.float32))
        return removed

    # -- matching ----------------------------------------------------------- #
    def match(self, embedding: np.ndarray, threshold: Optional[float] = None
              ) -> Tuple[str, float]:
        """Return ``(name, similarity)`` for the closest gallery entry."""
        limit = self.threshold if threshold is None else float(threshold)
        if len(self.names) == 0:
            return UNKNOWN, 0.0
        similarities = embedding_similarity(embedding, self.embeddings)[0]
        best = int(np.argmax(similarities))
        score = float(similarities[best])
        if score < limit:
            return UNKNOWN, score
        return self.names[best], score


class FaceRecognizer:
    """Detect, align, embed and identify faces in an image."""

    def __init__(self, detector_model: Optional[str] = None,
                 embedder_model: Optional[str] = None,
                 backend: str = "auto",
                 gallery: Optional[FaceGallery] = None,
                 threshold: float = DEFAULT_THRESHOLD,
                 detection_size: Tuple[int, int] = (640, 640),
                 embed_unknown: bool = False,
                 verbose: bool = False,
                 provider: str = "auto"):
        detector_path = find_face_model("detector", detector_model)
        if detector_path is None:
            raise FileNotFoundError(
                "no face detection model found; run `yolo-pose models pull --face`"
            )
        embedder_path = find_face_model("embedder", embedder_model)
        if embedder_path is None and (gallery is not None or embedder_model):
            raise FileNotFoundError(
                "no face embedding model found; run `yolo-pose models pull --face`"
            )

        self.detector = FaceDetector(detector_path, backend=backend,
                                     input_size=detection_size, verbose=verbose,
                                     provider=provider)
        self.embedder = (FaceEmbedder(embedder_path, backend=backend,
                                      verbose=verbose, provider=provider)
                         if embedder_path else None)
        self.gallery = gallery if gallery is not None else FaceGallery(threshold=threshold)
        self.threshold = float(threshold)
        self.embed_unknown = embed_unknown

    # -- inference ---------------------------------------------------------- #
    def recognize(self, image: np.ndarray,
                  score_threshold: Optional[float] = None) -> FaceResult:
        """Run the full pipeline on an RGB ``uint8`` image."""
        import time

        start = time.perf_counter()
        faces = self.detector.detect(image, score_threshold=score_threshold)
        recognized: List[RecognizedFace] = []

        for face in faces:
            entry = RecognizedFace(face=face)
            if self.embedder is not None:
                crop = align_face(image, face)
                embedding = self.embedder.embed(crop)
                if self.embed_unknown or len(self.gallery):
                    entry.embedding = embedding
                name, similarity = self.gallery.match(embedding, self.threshold)
                entry.name, entry.similarity = name, similarity
            recognized.append(entry)

        return FaceResult(
            faces=recognized,
            detector_backend=self.detector.backend_name,
            embedder_backend=self.embedder.backend_name if self.embedder else "none",
            inference_seconds=time.perf_counter() - start,
        )

    def enroll(self, image: np.ndarray, name: str,
               score_threshold: Optional[float] = None,
               min_face_size: int = 40) -> int:
        """Add every detected face in ``image`` to the gallery under ``name``."""
        if self.embedder is None:
            raise RuntimeError("enrolment needs a face embedding model")
        faces = self.detector.detect(image, score_threshold=score_threshold)
        added = 0
        for face in faces:
            width = face.box[2] - face.box[0]
            height = face.box[3] - face.box[1]
            if min(width, height) < min_face_size:
                continue
            crop = align_face(image, face)
            self.gallery.add(name, self.embedder.embed(crop))
            added += 1
        return added

    def describe(self) -> str:
        lines = [self.detector.describe()]
        if self.embedder:
            lines.append(self.embedder.describe())
        lines.append(f"gallery       : {len(self.gallery)} entries, "
                     f"{len(self.gallery.people)} people")
        lines.append(f"threshold     : {self.gallery.threshold}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

_FACE_COLOURS = (
    (80, 220, 120), (80, 170, 255), (255, 170, 60), (230, 90, 200),
    (120, 230, 230), (250, 220, 80), (170, 130, 255), (255, 120, 120),
)


def draw_faces(canvas: np.ndarray, result: FaceResult, threshold: float = DEFAULT_THRESHOLD,
               show_landmarks: bool = True, scale: int = 2) -> None:
    """Annotate a canvas with face boxes, identities and landmarks."""
    from ..render.skeleton import draw_circle, draw_label, draw_line, draw_rectangle

    palette: Dict[str, Tuple[int, int, int]] = {}
    for entry in result.faces:
        if entry.name not in palette:
            palette[entry.name] = (
                (255, 255, 255) if entry.name == UNKNOWN
                else _FACE_COLOURS[len(palette) % len(_FACE_COLOURS)])
        colour = palette[entry.name]

        draw_rectangle(canvas, entry.box, colour, width=2.0, alpha=0.95)
        label = (f"{entry.name} {entry.similarity:.2f}"
                 if entry.name != UNKNOWN else
                 f"{UNKNOWN} {entry.score:.2f}")
        draw_label(canvas, label, (entry.box[0], max(14, entry.box[1] - 16)),
                   (255, 255, 255), scale=scale, background=colour, alpha=0.75)

        if show_landmarks:
            for x, y in entry.face.landmarks:
                draw_circle(canvas, (x, y), 2.0, (255, 255, 255), fill=True, alpha=0.95)
