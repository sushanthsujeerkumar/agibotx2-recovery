from setuptools import find_packages, setup

package_name = 'x2_recovery_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/recovery.launch.py']),
    ],
    install_requires=['setuptools'],
    tests_require=['pytest'],
    zip_safe=True,
    maintainer='HRS X2 Recovery contributors',
    maintainer_email='maintainer@example.com',
    description='ROS 2 service and telemetry connected to real X2 recovery simulation.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'recovery_node = x2_recovery_ros.recovery_node:main',
        'telemetry_node = x2_recovery_ros.telemetry_node:main',
    ]},
)
