"""Face pipeline tests: SCRFD decoding, Umeyama alignment, gallery matching."""

import numpy as np
import pytest

from yolopose.face.align import (ARCFACE_TEMPLATE, embedding_similarity,
                                 estimate_similarity, warp_face)
from yolopose.face.recognizer import UNKNOWN, FaceGallery
from yolopose.face.scrfd import Face, _anchor_centers, _distance2bbox, decode_scrfd, nms


def test_arcface_template_shape():
    assert ARCFACE_TEMPLATE.shape == (5, 2)
    # Eyes are level in the canonical pose.
    assert ARCFACE_TEMPLATE[0][1] == pytest.approx(ARCFACE_TEMPLATE[1][1], abs=1.0)


def test_estimate_similarity_recovers_known_transform():
    source = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    angle = np.deg2rad(30.0)
    rotation = np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]], dtype=np.float32)
    scale = 2.0
    translation = np.array([5.0, -3.0], dtype=np.float32)
    destination = (scale * (rotation @ source.T).T) + translation

    matrix = estimate_similarity(source, destination)
    predicted = (matrix[:, :2] @ source.T).T + matrix[:, 2]
    assert np.allclose(predicted, destination, atol=1e-3)


def test_estimate_similarity_rejects_bad_shape():
    with pytest.raises(ValueError):
        estimate_similarity(np.zeros((3, 2)), np.zeros((4, 2)))


def test_warp_face_identity_matrix_preserves_image():
    image = np.zeros((112, 112, 3), dtype=np.uint8)
    image[20:80, 30:90] = (10, 200, 30)
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    warped = warp_face(image, matrix, (112, 112))
    assert warped.shape == image.shape
    assert np.abs(warped.astype(int) - image.astype(int)).max() <= 2


def test_anchor_centers_counts():
    centers = _anchor_centers(2, 3, 8)
    # 2 anchors per location, 6 locations
    assert centers.shape == (12, 2)
    assert centers[0].tolist() == [0.0, 0.0]
    assert centers[2].tolist() == [8.0, 0.0]


def test_distance2bbox():
    points = np.array([[10.0, 10.0]], dtype=np.float32)
    distances = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    box = _distance2bbox(points, distances)
    assert np.allclose(box, [[9.0, 8.0, 13.0, 14.0]])


def test_face_nms_suppresses_overlaps():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]],
                     dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms(boxes, scores, threshold=0.4)
    assert list(keep) == [0, 2]


def _fake_scrfd_outputs(input_size, scores_value=0.9):
    """Build synthetic SCRFD-style outputs for one anchor per level."""
    width, height = input_size
    outputs = {}
    for stride in (8, 16, 32):
        feature_h = (height + stride - 1) // stride
        feature_w = (width + stride - 1) // stride
        rows = feature_h * feature_w * 2
        outputs[f"score_{stride}"] = np.full((rows, 1), scores_value, dtype=np.float32)
        outputs[f"bbox_{stride}"] = np.full((rows, 4), 0.5, dtype=np.float32)
        outputs[f"kps_{stride}"] = np.full((rows, 10), 0.1, dtype=np.float32)
    return outputs


def test_decode_scrfd_produces_boxes():
    boxes, scores, landmarks = decode_scrfd(_fake_scrfd_outputs((640, 640)),
                                            (640, 640), score_threshold=0.5)
    assert boxes.shape[1] == 4
    assert landmarks.shape[1:] == (5, 2)
    assert boxes.shape[0] == scores.shape[0] == landmarks.shape[0]
    assert boxes.shape[0] > 0
    assert np.isfinite(boxes).all()


def test_decode_scrfd_threshold_filters_everything():
    boxes, scores, _ = decode_scrfd(_fake_scrfd_outputs((640, 640), scores_value=0.1),
                                    (640, 640), score_threshold=0.5)
    assert boxes.shape[0] == 0


def test_embedding_similarity_cosine():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    c = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    assert embedding_similarity(a, b)[0, 0] == pytest.approx(1.0)
    assert embedding_similarity(a, c)[0, 0] == pytest.approx(0.0, abs=1e-6)


def test_gallery_match_and_threshold():
    gallery = FaceGallery(threshold=0.5)
    gallery.add("alice", np.array([1.0, 0.0, 0.0], dtype=np.float32))
    gallery.add("bob", np.array([0.0, 1.0, 0.0], dtype=np.float32))

    name, score = gallery.match(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    assert name == "alice"
    assert score == pytest.approx(1.0)

    name, score = gallery.match(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert name == "bob"

    # An orthogonal vector matches nothing above the threshold.
    name, score = gallery.match(np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert name == UNKNOWN


def test_gallery_persistence_roundtrip(tmp_path):
    gallery = FaceGallery(threshold=0.4)
    gallery.add("alice", np.array([1.0, 0.0], dtype=np.float32))
    gallery.add("alice", np.array([0.9, 0.1], dtype=np.float32))
    gallery.add("bob", np.array([0.0, 1.0], dtype=np.float32))

    npz_path = tmp_path / "gallery.npz"
    gallery.save(str(npz_path))
    loaded = FaceGallery.load(str(npz_path))
    assert loaded.names == gallery.names
    assert np.allclose(loaded.embeddings, gallery.embeddings)
    # The default threshold applies when loading unless one is supplied.
    assert loaded.threshold == pytest.approx(0.35)
    assert FaceGallery.load(str(npz_path), threshold=0.4).threshold == pytest.approx(0.4)

    json_path = tmp_path / "gallery.json"
    gallery.save(str(json_path))
    loaded_json = FaceGallery.load(str(json_path))
    assert loaded_json.names == gallery.names
    assert np.allclose(loaded_json.embeddings, gallery.embeddings)


def test_gallery_requires_no_pickle(tmp_path):
    """Loading must not need pickle (object arrays would be a security issue)."""
    gallery = FaceGallery()
    gallery.add("alice", np.array([1.0, 0.0], dtype=np.float32))
    path = tmp_path / "gallery.npz"
    gallery.save(str(path))
    with np.load(str(path), allow_pickle=False) as data:
        assert "names_json" in data
        assert "embeddings" in data


def test_gallery_remove():
    gallery = FaceGallery()
    gallery.add("alice", np.array([1.0, 0.0], dtype=np.float32))
    gallery.add("bob", np.array([0.0, 1.0], dtype=np.float32))
    assert gallery.remove("alice") == 1
    assert gallery.names == ["bob"]
    assert gallery.embeddings.shape[0] == 1
