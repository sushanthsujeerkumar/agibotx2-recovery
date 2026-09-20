from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        ('controller', 'scripted', 'scripted, policy or reference_residual; reference is an external teacher plus trained correction'),
        ('checkpoint', '', 'Absolute policy checkpoint path'),
        ('vendor_assets', '', 'External vendor asset directory; required only for reference_residual'),
        ('render', 'false', 'Open a local MuJoCo viewer for the recovery episode'),
        ('seed', '0', 'Repeatable reset seed'),
        ('timeout_s', '60.0', 'Wall-clock deadline including simulator startup'),
        ('max_sim_duration_s', '20.0', 'Maximum simulated episode duration'),
        ('realtime', 'true', 'Pace simulation to wall time for a visible demonstration'),
        ('physics_profile', 'legacy', 'Versioned physics: legacy or guarded_v2'),
        ('log_period_s', '1.0', 'Telemetry log interval'),
    ]
    declarations = [DeclareLaunchArgument(name, default_value=value, description=description)
                    for name, value, description in arguments]
    parameters = {name: ParameterValue(LaunchConfiguration(name), value_type=kind)
                  for name, kind in {
                      'controller': str, 'checkpoint': str, 'vendor_assets': str, 'render': bool,
                      'seed': int, 'timeout_s': float, 'max_sim_duration_s': float,
                      'realtime': bool,
                      'physics_profile': str,
                  }.items()}
    return LaunchDescription(declarations + [
        Node(package='x2_recovery_ros', executable='recovery_node',
             name='x2_recovery', output='screen', parameters=[parameters]),
        Node(package='x2_recovery_ros', executable='telemetry_node',
             name='x2_telemetry', output='screen',
             parameters=[{'log_period_s': ParameterValue(LaunchConfiguration('log_period_s'), value_type=float)}]),
    ])
