"""Face detection and recognition (SCRFD + ArcFace-style embeddings)."""

from .align import (ARCFACE_TEMPLATE, FaceEmbedder, align_face,
                    embedding_similarity, estimate_similarity, warp_face)
from .recognizer import (FaceGallery, FaceRecognizer, FaceResult,
                         RecognizedFace, UNKNOWN, draw_faces)
from .scrfd import Face, FaceDetector, decode_scrfd

__all__ = [
    "Face", "FaceDetector", "decode_scrfd",
    "ARCFACE_TEMPLATE", "estimate_similarity", "warp_face", "align_face",
    "FaceEmbedder", "embedding_similarity",
    "FaceGallery", "FaceRecognizer", "FaceResult", "RecognizedFace",
    "UNKNOWN", "draw_faces",
]
