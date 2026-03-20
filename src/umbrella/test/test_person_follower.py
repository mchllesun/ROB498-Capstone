"""Unit tests for the pure helper functions in person_follower_node.

Run with:  python3 -m pytest test/test_person_follower.py -v
(from the umbrella package directory)

These tests do NOT require a running ROS system -- they exercise only the
pure-Python math that converts detections into position steps.
"""

import math
import sys
import os

import pytest

# Add the scripts directory so we can import the module directly.
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), '..', 'scripts'),
)

from follower_math import compute_position_step, select_best_person


# ---------------------------------------------------------------------------
# Helpers to build fake Detection2D-like objects without importing ROS msgs
# ---------------------------------------------------------------------------

class _Point:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y


class _Center:
    def __init__(self, x=0.0, y=0.0):
        self.position = _Point(x, y)


class _BBox:
    def __init__(self, cx, cy, w, h):
        self.center = _Center(cx, cy)
        self.size_x = w
        self.size_y = h


class _Hypothesis:
    def __init__(self, class_id, score):
        self.class_id = class_id
        self.score = score


class _HypWithPose:
    def __init__(self, class_id, score):
        self.hypothesis = _Hypothesis(class_id, score)


class _Detection:
    def __init__(self, cx, cy, w, h, class_id, score):
        self.bbox = _BBox(cx, cy, w, h)
        self.results = [_HypWithPose(class_id, score)]


# ===================================================================
# Tests for compute_position_step
# ===================================================================

class TestComputePositionStep:
    """Verify the IBVS pixel-error-to-step conversion."""

    IMG_W = 640.0
    IMG_H = 640.0
    KP = 0.6
    DEADBAND = 0.08
    MAX_STEP = 0.15

    def _step(self, cx, cy):
        return compute_position_step(
            cx, cy, self.IMG_W, self.IMG_H,
            self.KP, self.KP, self.DEADBAND, self.MAX_STEP,
        )

    def test_center_gives_zero_step(self):
        sx, sy = self._step(320.0, 320.0)
        assert sx == 0.0
        assert sy == 0.0

    def test_deadband_suppresses_small_error(self):
        """A person slightly off-center (< deadband) should produce zero step."""
        offset = self.DEADBAND * 0.5 * (self.IMG_W / 2.0)
        sx, sy = self._step(320.0 + offset, 320.0 + offset)
        assert sx == 0.0
        assert sy == 0.0

    def test_right_of_center_gives_positive_step_y(self):
        """Person on the right side of the image -> positive step_y."""
        sx, sy = self._step(480.0, 320.0)
        assert sy > 0.0
        assert sx == 0.0  # no forward/back since cy is centered

    def test_left_of_center_gives_negative_step_y(self):
        sx, sy = self._step(160.0, 320.0)
        assert sy < 0.0
        assert sx == 0.0

    def test_below_center_gives_positive_step_x(self):
        """Person below image center -> positive step_x (forward)."""
        sx, sy = self._step(320.0, 480.0)
        assert sx > 0.0
        assert sy == 0.0

    def test_above_center_gives_negative_step_x(self):
        sx, sy = self._step(320.0, 160.0)
        assert sx < 0.0
        assert sy == 0.0

    def test_step_is_clamped(self):
        """Even with the person at the very edge, step must not exceed max."""
        sx, sy = self._step(640.0, 640.0)
        assert abs(sx) <= self.MAX_STEP + 1e-9
        assert abs(sy) <= self.MAX_STEP + 1e-9

    def test_proportional_scaling(self):
        """Doubling the error should double the step (using small offsets to avoid clamp)."""
        _, sy1 = self._step(350.0, 320.0)  # 30px right of center
        _, sy2 = self._step(380.0, 320.0)  # 60px right of center
        assert pytest.approx(sy2, rel=0.01) == sy1 * 2.0


# ===================================================================
# Tests for select_best_person
# ===================================================================

class TestSelectBestPerson:
    def test_no_detections_returns_none(self):
        assert select_best_person([]) is None

    def test_single_person(self):
        det = _Detection(100.0, 200.0, 50.0, 80.0, 'person', 0.9)
        result = select_best_person([det])
        assert result is not None
        cx, cy, w, h, conf = result
        assert cx == 100.0
        assert cy == 200.0
        assert conf == 0.9

    def test_ignores_non_person(self):
        det = _Detection(100.0, 200.0, 50.0, 80.0, 'car', 0.95)
        assert select_best_person([det]) is None

    def test_ignores_low_confidence(self):
        det = _Detection(100.0, 200.0, 50.0, 80.0, 'person', 0.1)
        assert select_best_person([det], min_confidence=0.35) is None

    def test_picks_largest_box(self):
        small = _Detection(100.0, 100.0, 30.0, 30.0, 'person', 0.8)
        large = _Detection(200.0, 200.0, 100.0, 100.0, 'person', 0.7)
        result = select_best_person([small, large])
        assert result is not None
        cx, cy, w, h, conf = result
        assert cx == 200.0  # the larger box
        assert conf == 0.7

    def test_filters_mixed_classes(self):
        car = _Detection(100.0, 100.0, 200.0, 200.0, 'car', 0.99)
        person = _Detection(300.0, 300.0, 50.0, 50.0, 'person', 0.5)
        result = select_best_person([car, person])
        assert result is not None
        assert result[0] == 300.0  # person bbox center

    def test_custom_target_class(self):
        det = _Detection(100.0, 200.0, 50.0, 80.0, 'dog', 0.85)
        assert select_best_person([det], target_class='dog') is not None
        assert select_best_person([det], target_class='person') is None
