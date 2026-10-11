"""Safe launch: motion and motor-driver startup are disabled by default."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    hw1 = os.path.join(get_package_share_directory('hw1_navigation'), 'hw1')
    return LaunchDescription([
        DeclareLaunchArgument('enable_motion', default_value='false'),
        DeclareLaunchArgument('enable_motor_driver', default_value='false'),
        Node(
            package='robot_control',
            executable='motor_control',
            name='motor_controller_node',
            output='screen',
            condition=IfCondition(LaunchConfiguration('enable_motor_driver')),
        ),
        Node(
            package='hw1_navigation',
            executable='waypoint_follower',
            name='hw1_waypoint_follower',
            output='screen',
            parameters=[{
                'enable_motion': ParameterValue(LaunchConfiguration('enable_motion'), value_type=bool),
                'waypoints_file': os.path.join(hw1, 'waypoints.txt'),
                'calibration_file': os.path.join(hw1, 'calibration.json'),
            }],
        ),
    ])
