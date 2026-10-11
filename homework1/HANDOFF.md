# CSE 276A HW1 — Progress Handoff

## Current status
Paused after Step 5.5.4: Rotation Calibration.
Step 5.5.5 has NOT been performed.

Branch: jing/hw1
Environment: Rubik Pi 3, Ubuntu 24.04, ROS2 Jazzy.

## Completed implementation
- ROS2 waypoint navigation package
- Differential/skid-steer kinematic model
- ALIGN / DRIVE / FINAL_ALIGN controller
- Gyroscope integration for heading estimation
- Offline simulation
- 9 unit tests passed
- ROS2 package build successful

## Hardware findings

IMU:
- /imu_data publishes approximately 48 Hz.
- Fused yaw is inaccurate: physical 90-degree rotation
  produced only ~14-17 degrees of fused yaw change.
- Gyroscope integration successfully measured ~90.9 degrees
  for a manual 90-degree rotation.
- The existing motor driver publishes gyro values in deg/s.

Motor tests:
- L=+0.15, R=+0.15, 0.7 s, wheels elevated:
  both sides rotated forward; stopped normally.
- L=+0.15, R=+0.15, 1.0 s, on floor:
  traveled ~0.16 m, approximately straight.
- Preliminary linear_gain: 1.067.
- L=-0.15, R=+0.15, 0.7 s:
  no observed stationary rotation.
- L=+0.10, R=+0.30, 1.0 s:
  moving left turn succeeded.
- Gyro-measured forward phase: +2.90 degrees.
- Gyro-measured moving-turn phase: +7.69 degrees.
- Preliminary moving-turn yaw_gain: ~0.671,
  ONLY for this command regime.
- L=-0.30, R=+0.30, 1.0 s:
  stationary-spin gyro angle = -0.22 degrees,
  effectively no rotation.

## Important blocker
Stationary rotation is not reliable, while moving
left turns work. Possible causes include friction,
insufficient torque, or motor reversal behavior.

Do NOT treat the stationary-spin result as a valid yaw_gain.
The current controller assumes a roughly linear yaw response
and may require modifications for skid-steer dead zones.

calibration.json intentionally retains null gains.
Autonomous navigation has NOT been validated on hardware.
DO NOT enable autonomous motion yet.

## Next step — NOT YET EXECUTED
Step 5.5.5:
Securely elevate all driven wheels and briefly test
L=-0.30, R=+0.30 under direct supervision.
Check whether left wheels reverse and right wheels advance.

After diagnosing stationary rotation:
1. Repeat motor calibration.
2. Update controller and calibration parameters.
3. Validate real waypoint navigation.
4. Record video and prepare HW1 report/submission.

## Useful commands
source /opt/ros/jazzy/setup.bash
source ~/jing_hw1_ws/install/setup.bash

ros2 run robot_control motor_control
ros2 topic info /motor_commands -v

cd ~/jing_hw1_ws/src/robotics-projects/homework1
python3 -m unittest discover -s tests -v
python3 -m hw1_navigation.simulate
