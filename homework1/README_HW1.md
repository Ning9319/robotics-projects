# CSE276A HW1 — Waypoint Navigation (initial implementation)

This package is independently developed under `homework1/`. It **does not modify** the professor's `robot_control` driver.

## Architecture

- `core.py`: differential-drive kinematics, IMU-assisted command dead reckoning, heading/position controller
- `waypoint_follower.py`: ROS2 subscription `/imu_data` (sensor_msgs/Imu), publication `/motor_commands` (std_msgs/Float32MultiArray `[left,right]`)
- `simulate.py`: deterministic *ideal-model* simulation; no real hardware
- `tests/test_navigation.py`: pure Python unit tests
- `launch/hw1.launch.py`: safe launch; motor driver and motion **off by default**

### Kinematic model

The professor's motor commands `u_L,u_R` are dimensionless, limited to `[-0.5,0.5]`. We model the *effective* wheel/ground speed using measured gains:

```
v = linear_gain * (u_L + u_R) / 2
omega = yaw_gain * (u_R - u_L)
xdot = v*cos(theta)
ydot = v*sin(theta)
thetadot = omega
```

`linear_gain` in m/s per command, `yaw_gain` in rad/s per command difference. These approximate a differential/skid-steer rover. They are **not** known until measured. Position is integrated using command-derived velocity and *measured IMU yaw*, so wheel slip and motor mismatch cause cumulative position drift. The first valid IMU yaw is heading zero, and the robot must be placed at waypoint `(0,0,0)` pointing along the chosen +X axis.

Controller: **ALIGN** toward the waypoint → **DRIVE** with heading corrections → **FINAL_ALIGN** at the waypoint's required angle. Both position and orientation tolerances must pass. Stops between waypoints.

## 1. No-hardware tests (safe)

```bash
cd ~/jing_hw1_ws/src/robotics-projects/homework1
python3 -m unittest discover -s tests -v
python3 -m hw1_navigation.simulate
```

The simulation uses **synthetic gains** (0.6 and 2.0) only to check code logic. The fact that it succeeds does not prove the actual Rover can follow the waypoints.

## 2. Build ROS2 on Rubik Pi

```bash
cd ~/jing_hw1_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select hw1_navigation --symlink-install
source install/setup.bash
ros2 pkg executables hw1_navigation
```

## 3. Default safe ROS2 launch

```bash
ros2 launch hw1_navigation hw1.launch.py
```

This only runs the disarmed follower and **does not start motor_control**.

## 4. Hardware prerequisites / calibration (NOT completed)

Before real movement, verify the physical motor-direction convention, IMU orientation/yaw sign and message freshness, flat clear floor, wheel speed response, and that `/dev/ttyUSB0` belongs to the rover controller. Check no other motor controller is running. Have someone ready to stop power.

Measure two separate physical parameters with short, supervised low-speed experiments:

1. Straight: symmetric `u_L=u_R=u` for duration `T`, measured travel distance `d` gives approximate `linear_gain=d/(u*T)`.
2. Rotation: `u_L=-u, u_R=+u`, measured **signed, unwrapped** yaw change `delta` gives `yaw_gain=abs(delta)/(2*u*T)`; verify `imu_yaw_sign` from physical turn direction. Avoid raw subtraction across ±π.

Populate `calibration.json` with measured positive gains. Default `null` gains intentionally block motion. Never copy the synthetic gains from simulation into the real calibration.

For a **later, supervised** physical test only, with calibration completed and an empty course-approved test area:

```bash
ros2 launch hw1_navigation hw1.launch.py enable_motor_driver:=true enable_motion:=true
```

This starts both the existing serial motor driver and follower. It can move the real Rover. Do not execute while calibrations remain unverified. The follower stops on stale IMU (>0.4 s), delayed timer (>0.2 s), per-waypoint timeout (60 s), completion, or shutdown. The professor's motor driver also stops after 0.15 s without commands. Stopping is software-based; retain physical emergency-stop access.

## 5. Report / evidence pending

Calibrate, physically run all seven waypoints, record actual deviations and limitations, capture an unlisted YouTube video, then write report and submit ZIP. **No physical results have been measured yet.**
