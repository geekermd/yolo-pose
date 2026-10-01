"""Pose decoding, joint angles and pipeline tests."""

import numpy as np
import pytest

from yolopose.vision.imaging import (LetterboxInfo, letterbox, map_box,
                                     resize_bilinear, save_image, load_image,
                                     to_network_input)
from yolopose.vision.joints import angle_at, compute_joint_angles
from yolopose.vision.pose import (KEYPOINT_NAMES, Pose, box_iou, decode_poses,
                                  nms, xywh_to_xyxy)


def test_keypoint_vocabulary():
    assert len(KEYPOINT_NAMES) == 17
    assert KEYPOINT_NAMES[0] == "nose"
    assert KEYPOINT_NAMES[-1] == "right_ankle"


def test_letterbox_preserves_aspect_ratio():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    padded, info = letterbox(image, (640, 640))
    assert padded.shape == (640, 640, 3)
    assert info.scale == pytest.approx(1.0)
    assert info.pad_y == 80
    assert info.pad_x == 0

    wide, info = letterbox(np.zeros((240, 480, 3), dtype=np.uint8), (640, 640))
    assert wide.shape == (640, 640, 3)
    assert info.pad_y == 160
    assert info.pad_x == 0


def test_map_box_inverts_letterbox():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[100:300, 200:400] = 255
    padded, info = letterbox(image, (640, 640))
    scale = info.scale
    network_box = np.array([200 * scale + info.pad_x, 100 * scale + info.pad_y,
                            400 * scale + info.pad_x, 300 * scale + info.pad_y],
                           dtype=np.float64)
    mapped = map_box(network_box, info)
    assert np.allclose(mapped, [200, 100, 400, 300], atol=1.0)


def test_resize_bilinear_identity_and_scaling():
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    assert np.array_equal(resize_bilinear(image, (4, 4)), image)
    doubled = resize_bilinear(image, (8, 8))
    assert doubled.shape == (8, 8, 3)
    # Pixel-centre alignment preserves the overall range and stays close to the
    # nearest-neighbour replication (exact equality is not guaranteed at edges).
    assert int(doubled.min()) >= int(image.min()) - 1
    assert int(doubled.max()) <= int(image.max()) + 1
    assert np.abs(doubled.astype(int) - image.repeat(2, 0).repeat(2, 1)).mean() <= 4

    shrunk = resize_bilinear(image, (2, 2))
    assert shrunk.shape == (2, 2, 3)


def test_to_network_input_normalisation():
    image = np.full((8, 8, 3), 255, dtype=np.uint8)
    tensor = to_network_input(image)
    assert tensor.shape == (1, 3, 8, 8)
    assert tensor.dtype == np.float32
    assert tensor.max() == pytest.approx(1.0)


def test_xywh_to_xyxy():
    boxes = np.array([[10, 10, 4, 6]], dtype=np.float32)
    assert np.allclose(xywh_to_xyxy(boxes), [[8, 7, 12, 13]])


def test_box_iou_and_nms():
    a = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert box_iou(a[:1], a[1:2])[0, 0] == pytest.approx(1.0)
    keep = nms(a, scores, iou_threshold=0.5)
    assert list(keep) == [0, 2]


def test_decode_poses_shapes_and_threshold():
    info = LetterboxInfo(scale=1.0, pad_x=0.0, pad_y=0.0, src_width=640,
                         src_height=640, net_width=640, net_height=640)
    channels, anchors = 56, 3
    output = np.zeros((1, channels, anchors), dtype=np.float32)
    for index in range(anchors):
        output[0, 0, index] = 100 + index * 200   # cx
        output[0, 1, index] = 100                # cy
        output[0, 2, index] = 40                 # w
        output[0, 3, index] = 60                 # h
        output[0, 4, index] = 0.9 - index * 0.1  # score
        output[0, 5:5 + 51, index] = 0.5
    poses = decode_poses(output, info, conf_threshold=0.5)
    assert len(poses) == 3
    assert poses[0].keypoints.shape == (17, 2)
    assert poses[0].keypoint_scores.shape == (17,)
    assert poses[0].box[2] - poses[0].box[0] == pytest.approx(40, abs=1)


def test_decode_poses_respects_confidence():
    info = LetterboxInfo(1.0, 0.0, 0.0, 640, 640, 640, 640)
    output = np.zeros((1, 56, 5), dtype=np.float32)
    output[0, 0] = [50, 150, 250, 350, 450]   # cx, well separated
    output[0, 1] = [50, 150, 250, 350, 450]   # cy
    output[0, 2] = 20.0                       # w
    output[0, 3] = 20.0                       # h
    output[0, 4] = [0.1, 0.2, 0.3, 0.4, 0.5]
    output[0, 5:5 + 51] = 0.5                 # visible keypoints
    poses = decode_poses(output, info, conf_threshold=0.35)
    # Two detections clear the threshold, and they do not overlap.
    assert len(poses) == 2
    assert all(p.score >= 0.35 for p in poses)
    # The strongest detection comes first.
    assert poses[0].score >= poses[1].score


def test_decode_poses_suppresses_duplicate_boxes():
    info = LetterboxInfo(1.0, 0.0, 0.0, 640, 640, 640, 640)
    output = np.zeros((1, 56, 2), dtype=np.float32)
    output[0, 0] = 100.0
    output[0, 1] = 100.0
    output[0, 2] = 40.0
    output[0, 3] = 40.0
    output[0, 4] = [0.9, 0.8]      # identical boxes -> NMS must collapse them
    poses = decode_poses(output, info, conf_threshold=0.5)
    assert len(poses) == 1
    assert poses[0].score == pytest.approx(0.9)


def test_decode_poses_empty_when_all_below_threshold():
    info = LetterboxInfo(1.0, 0.0, 0.0, 640, 640, 640, 640)
    output = np.zeros((1, 56, 4), dtype=np.float32)
    output[0, 2] = 20.0
    output[0, 3] = 20.0
    assert decode_poses(output, info, conf_threshold=0.5) == []


def test_angle_at_known_values():
    assert angle_at((0, 0), (1, 0), (1, 1)) == pytest.approx(45.0)
    assert angle_at((0, 0), (1, 0), (-1, 0)) == pytest.approx(180.0)
    assert angle_at((0, 0), (1, 0), (0, 1)) == pytest.approx(90.0)


def _straight_pose() -> Pose:
    """A synthetic person standing with arms down; angles are easy to verify."""
    keypoints = np.zeros((17, 2), dtype=np.float32)
    layout = {
        "nose": (0, 0), "left_eye": (-3, -3), "right_eye": (3, -3),
        "left_ear": (-6, 0), "right_ear": (6, 0),
        "left_shoulder": (-20, 20), "right_shoulder": (20, 20),
        "left_elbow": (-20, 60), "right_elbow": (20, 60),
        "left_wrist": (-20, 100), "right_wrist": (20, 100),
        "left_hip": (-15, 100), "right_hip": (15, 100),
        "left_knee": (-15, 160), "right_knee": (15, 160),
        "left_ankle": (-15, 220), "right_ankle": (15, 220),
    }
    for name, (x, y) in layout.items():
        index = KEYPOINT_NAMES.index(name)
        keypoints[index] = (x + 100, y + 50)
    return Pose(box=np.array([60, 20, 140, 290], dtype=np.float32), score=0.95,
                keypoints=keypoints,
                keypoint_scores=np.ones(17, dtype=np.float32))


def test_joint_angles_straight_limbs():
    pose = _straight_pose()
    result = compute_joint_angles(pose)
    # Arms hang straight down and legs are straight, so the limb joints are
    # fully extended.  The measured joint is the vertex of the angle.
    assert result["right_elbow"] == pytest.approx(180.0, abs=1.0)
    assert result["left_elbow"] == pytest.approx(180.0, abs=1.0)
    assert result["left_knee"] == pytest.approx(180.0, abs=1.0)
    assert result["right_knee"] == pytest.approx(180.0, abs=1.0)
    # Shoulder: the arm is perpendicular to the shoulder axis (90 degrees).
    assert result["right_shoulder"] == pytest.approx(90.0, abs=2.0)
    # Every joint in the vocabulary is measurable on a full-body pose.
    assert len(result.angles) == 12
    assert result.skipped == {}


def test_joint_angles_skip_low_confidence():
    pose = _straight_pose()
    pose.keypoint_scores[KEYPOINT_NAMES.index("left_wrist")] = 0.1
    result = compute_joint_angles(pose, keypoint_threshold=0.5)
    # The left elbow needs the wrist, so it drops out; the right side is intact.
    assert "left_elbow" not in result.angles
    assert "left_elbow" in result.skipped
    assert "low confidence" in result.skipped["left_elbow"]
    assert "right_elbow" in result.angles
    # The right side is unaffected.
    assert "right_elbow" in result.angles


def test_joint_angles_bent_elbow():
    pose = _straight_pose()
    elbow = KEYPOINT_NAMES.index("left_elbow")
    pose.keypoints[KEYPOINT_NAMES.index("left_wrist")] = (
        pose.keypoints[elbow][0] - 40, pose.keypoints[elbow][1])
    result = compute_joint_angles(pose)
    assert "left_elbow" in result.angles
    # Forearm now points sideways while the upper arm points down: 90 degrees.
    assert result["left_elbow"] == pytest.approx(90.0, abs=1.0)


def test_pose_as_dict_has_all_keypoints():
    pose = _straight_pose()
    payload = pose.as_dict()
    assert set(payload["keypoints"]) == set(KEYPOINT_NAMES)
    assert "score" in payload


def test_image_roundtrip_png(tmp_path):
    image = np.zeros((16, 24, 3), dtype=np.uint8)
    image[4:10, 6:18] = (200, 100, 50)
    path = tmp_path / "roundtrip.png"
    save_image(str(path), image)
    loaded = load_image(str(path))
    assert loaded.shape == image.shape
    assert np.array_equal(loaded, image)


def test_image_roundtrip_ppm(tmp_path):
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    path = tmp_path / "roundtrip.ppm"
    save_image(str(path), image)
    loaded = load_image(str(path))
    assert np.array_equal(loaded, image)
