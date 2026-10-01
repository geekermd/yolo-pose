"""yolo-pose — human joint (pose) recognition with YOLO on a pure-NumPy ONNX runtime.

The inference path has no compiled dependency: ONNX graphs are parsed by
:mod:`yolopose.core.onnx_graph` and executed by :mod:`yolopose.core.runtime`,
so the same code runs on x86-64, aarch64 and other NumPy platforms.  ONNX
Runtime is used automatically when it is installed, for extra speed.
"""

__version__ = "1.0.0"
__license__ = "MIT"
__author__ = "geekmd"

from .vision.detector import Detector, DetectionResult  # noqa: F401
from .vision.pose import KEYPOINT_NAMES, Pose, SKELETON_EDGES  # noqa: F401
from .vision.joints import JointAngles, compute_joint_angles  # noqa: F401

__all__ = [
    "__version__",
    "Detector",
    "DetectionResult",
    "Pose",
    "KEYPOINT_NAMES",
    "SKELETON_EDGES",
    "JointAngles",
    "compute_joint_angles",
]
