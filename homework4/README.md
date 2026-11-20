# CSE 276A HW4 — measured map, minimum files

All measurements in `config.yaml` are the student's 2026-11-19 physical setup.
Measured does **not** mean vision/localization, motor, or trajectory are validated.

## 1. What's in this folder

- `config.yaml`: eight measured perimeter tags, 55 cm center obstacle, start/goal, robot footprint.
- `tags.yaml`: HW2's detector parameters with the **eight** HW4 tag IDs and the `tag_#` TF frame names expected by the HW2 camera-to-world transform.
- `geometry.py`, `planning.py`: hand-written C-space geometry + visibility/Dijkstra shortest path and widest-clearance grid path.
- `plan.py`: offline graph / clearance / comparison figures, JSON planned paths.
- `navigate.py`: live known-map navigation using existing HW2 `/tf` transform, HW1 calibration and motor publishing; no new detector, motor driver, or SLAM.

`output/` is produced automatically. Do not commit CSV logs/videos or plots unless needed for the report.

## 2. Geometric assumptions

Field: 2.44 m square, world +x along tags 9,8 and +y along 1,2.
Obstacle x/y in [0.945,1.495], height 0.455 m (height not used in 2D planning).
Robot body 0.175 m x 0.170 m; approximate rotating robot by **circumscribed disk r=0.123 m**.
Planning inflation = 0.123 + **0.100 m extra safety margin** = 0.223 m.
This 10 cm is an INITIAL conservative tuning value, not a measured localization guarantee.
The landmark height z=0.14 m, size=0.165 m, camera longitudinal offset=+0.09 m in the existing HW2 `check_localization.py` and **must not be reapplied**.

The shortest result is optimal only for the **conservatively inflated axis-aligned rectangle model**. The safety result is maximum bottleneck raw clearance **on its 5 cm grid**, not exact continuous Voronoi. Both are collision-checked offline. Report plot and metrics are planned, not executed.

## 3. Offline before robot motion

Inside the rover repository root `~/ros2_ws/src/robotics-projects`:

```bash
python3 homework4/plan.py
python3 homework4/navigate.py --mode shortest   # no --drive, motors disabled
```

Offline plotting requires `python3-yaml`, `matplotlib`; navigation requires pre-existing `rclpy`, `numpy`, HW1, HW2 and the ROS stack. The command must run at repo root because the HW1/HW2 folders remain there.

Inspect `homework4/output/config_plans.json` and `config_comparison.png`. Do not drive until the observed robot position and heading at the Start agree with (2.02,0.42,90deg), all tag IDs are correct and the physical corridor is free.

## 4. Four PowerShell windows (SSH to the rover)

In **each** Windows PowerShell:

```powershell
ssh ubuntu@100.64.197.241
```

Then, in the connected **Ubuntu shell** for every window:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
cd ~/ros2_ws/src/robotics-projects
```

- **PS1 camera**: `ros2 run robot_vision_camera robot_vision_camera` (or the camera launch used successfully in HW2/HW3).
- **PS2 rectify + detector** (run both in this Ubuntu shell):

  ```bash
  ros2 run image_proc rectify_node --ros-args -r image:=/camera/image_raw -r camera_info:=/camera/camera_info -r image_rect:=/camera/image_rect &
  RECT_PID=$!
  trap 'kill "$RECT_PID" 2>/dev/null' EXIT
  ros2 run apriltag_ros apriltag_node --ros-args --params-file "$PWD/homework4/tags.yaml" -r image_rect:=/camera/image_rect -r camera_info:=/camera/camera_info
  ```

- **PS3 motor driver**: `ros2 run robot_control motor_control` (do not start keyboard / HW1 / HW2 control concurrently).
- **PS4 inspect + preview**: `ros2 topic list; ros2 topic echo /tf --once` and `python3 homework4/navigate.py --mode shortest` (no wheel motion). Stop preview with Ctrl+C.

**Check actual topic names** first: `/camera/image_rect`, `/camera/camera_info`, `/tf`, `/motor_commands` must match installed nodes. If camera launch *already* publishes correctly rectified images, don't start another rectifier; run only the detector in PS2. If detector's TF is `tag36h11:<id>` instead of `tag_<id>`, `tags.yaml` was not applied correctly.

## 5. Controlled run sequence (ONLY when ready)

After a no-motion pose check, clear floor, charged battery, external emergency stop/operator:

```bash
python3 homework4/navigate.py --mode shortest --drive --stop-after-wp 1
python3 homework4/navigate.py --mode shortest --drive
# Place rover at the marked Start again before testing the other mode.
python3 homework4/navigate.py --mode safety --drive --stop-after-wp 1
python3 homework4/navigate.py --mode safety --drive
```

Run commands **one at a time**, physically reset to marked start and 90-degree yaw for each attempt, and verify motor-stop behavior first with wheels safely raised. `--drive` is the ONLY flag that enables motor commands. Each route does a 2-second Start-pose check, follows waypoints, and writes a separate CSV.

If visual localization stops updating for >0.75 s: stop immediately, then attempt **one bounded in-place left/right sweep** with stop intervals and two newly accepted TF poses required for resuming. No start-up blind spin, no blind forward drive, no infinite sweeping; stop permanently if search fails, known obstacle/wall clearance is insufficient, or pose deviates too far from the planned corridor. Camera pose fusion/turn timing are still not physically verified for HW4; keep someone next to the rover and be ready to halt manually.

## 6. Upload to your GitHub repository (not uploaded by this ZIP)

Copy **the `homework4/` folder**, excluding `output/`, into the repo in Windows or on the rover. From repository root:

```bash
git switch -c jing/hw4-path-planning
git add homework4/config.yaml homework4/tags.yaml homework4/geometry.py homework4/planning.py homework4/plan.py homework4/navigate.py homework4/README.md
git commit -m "Implement measured HW4 path planners and tag-search recovery"
git push -u origin jing/hw4-path-planning
```

Review the diff before opening a PR or merging to main. This package does **not** change HW1/HW2/HW3 files.

## 7. Still needed for submission

Two independent physical runs/videos, per-run CSV trajectory comparison, and a PDF report explaining each representation, algorithm, visualizations, results, and architecture. Do not report the planned lengths as measured executed distances.
