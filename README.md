# ROB498-Capstone

## Prototype person-follow setup

This workspace now includes a prototype ROS 2 node for a downward-facing camera use case:

- `flight_exercise_3/scripts/person_follow_yolo.py`
- `flight_exercise_3/launch/person_follow.launch.py`

The node subscribes to a camera image, runs a YOLO model, selects the largest `person`
detection, and publishes local position setpoints that try to keep that person centered
under the drone while holding a fixed hover height.

### Assumptions

- The camera is mounted facing downward with a roughly consistent alignment to the drone body.
- `mavros` local position is available on `/mavros/local_position/pose`.
- Camera images are available on `/imx/image/raw`.
- `cv_bridge`, OpenCV, and `ultralytics` are installed in the runtime environment.
- The YOLO weights file `yolo26n.pt` is available where Ultralytics can load it.

### Launch

Build the workspace and run:

```bash
ros2 launch flight_exercise_3 person_follow.launch.py
```

The launch file defaults to `yolo26n.pt` with FP16 enabled on `cuda:0`.
Inference image size defaults to `320x320`.

### Important prototype note

This is a vision-centering prototype, not a production-safe follower. The X/Y mapping from
image error to local movement is intentionally simple and will need tuning, rate limiting,
and safety gating before flight testing on hardware.
