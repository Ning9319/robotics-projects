from glob import glob
from setuptools import find_packages, setup

package_name = 'hw1_navigation'
setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/hw1', ['waypoints.txt', 'calibration.json']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CSE276A team',
    maintainer_email='noreply@example.com',
    description='CSE276A HW1 differential-drive waypoint follower',
    license='MIT',
    entry_points={
        'console_scripts': [
            'waypoint_follower = hw1_navigation.waypoint_follower:main',
            'simulate_hw1 = hw1_navigation.simulate:main',
        ],
    },
)
