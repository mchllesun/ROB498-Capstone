"""Pure helper functions for person-following control logic.

No ROS dependencies -- safe to import anywhere and unit-test directly.
"""

from typing import Optional, Tuple


def select_best_person(
    detections,
    target_class: str = 'person',
    min_confidence: float = 0.35,
) -> Optional[Tuple[float, float, float, float, float]]:
    """Pick the largest bounding box matching *target_class*.

    *detections* must be an iterable of objects with:
      - bbox.center.position.x / .y, bbox.size_x, bbox.size_y
      - results[0].hypothesis.class_id, results[0].hypothesis.score

    Returns (center_x, center_y, width, height, confidence) or None.
    """
    best = None
    best_area = -1.0

    for det in detections:
        if not det.results:
            continue
        hyp = det.results[0]
        if hyp.hypothesis.class_id != target_class:
            continue
        if hyp.hypothesis.score < min_confidence:
            continue

        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        w = det.bbox.size_x
        h = det.bbox.size_y
        area = w * h
        if area > best_area:
            best_area = area
            best = (cx, cy, w, h, hyp.hypothesis.score)

    return best


def compute_position_step(
    center_x: float,
    center_y: float,
    img_w: float,
    img_h: float,
    kp_x: float,
    kp_y: float,
    deadband: float,
    max_step: float,
) -> Tuple[float, float]:
    """Convert a bounding-box center (pixels) into an XY position step (metres).

    Returns (step_x, step_y) in the drone's local frame where:
      - step_x corresponds to forward/back (mapped from image Y axis)
      - step_y corresponds to left/right   (mapped from image X axis)
    """
    norm_err_x = (center_x - img_w / 2.0) / (img_w / 2.0)
    norm_err_y = (center_y - img_h / 2.0) / (img_h / 2.0)

    step_x = 0.0 if abs(norm_err_y) < deadband else norm_err_y * kp_x
    step_y = 0.0 if abs(norm_err_x) < deadband else norm_err_x * kp_y

    step_x = max(-max_step, min(max_step, step_x))
    step_y = max(-max_step, min(max_step, step_y))

    return step_x, step_y
