# Assessment requirements and evidence

This maps the official HRS take-home requirements to the delivered partial-result project. Actual joint position/speed compliance remains unresolved and is explicitly identified below. The original recovery criterion and the subsequently added clean-stance criterion are reported separately. The posture limitation remains explicit.

| Requirement | Implementation or recorded evidence |
|---|---|
| AgiBot X2 URDF, simulator and model changes | [Model documentation](model.md); source URDF, vendor license, metadata and meshes in `assets/x2` |
| Floating base, flat floor and collisions | `assets/x2/scene.xml`, `src/x2_recovery/common.py`, [model validation](../assets/x2/model_validation.json) |
| Respect joint and actuator limits | **Partially satisfied / unresolved:** URDF ranges and command effort bounds are configured, but actual state excursions occur in all five retained-policy trials. [Physics-step audit](../artifacts/validation/final_reproduction/physics_step_limit_audit.json). No fully limit-compliant recovery claim. |
| Supine start with contact settling | `reset_cpu` in `common.py`; one second of controlled settling before episode time starts. Small numerical soft-contact penetration is recorded, not hidden. |
| Observations, actions, reward and termination | [Environment specification](environment.md), `env.py`, `common.py` |
| Actual RL experiment, saved policy and reward plot | PPO configs, full checkpoints, CPU actors, progress logs and `training_curve.png` under each `artifacts/submission/` run |
| Two ROS nodes, colcon, one launch file | `ros2_ws/src/x2_recovery_ros`; `recovery.launch.py` launches recovery and telemetry nodes |
| Trigger response before episode; busy rejection | Queue reservation and simulator subprocess; [real integration logs](../ros2_ws/validation/policy_recovery/integration.json) and supervisor unit tests |
| Actual simulator joint names, positions and timestamps | Worker reads `RecoveryRuntime` state after each control step; integration logs show 31 changing joints and increasing simulation timestamps |
| Required topics and statuses | `/x2/recovery_status`, `/x2/joint_states`; `IDLE`, `RUNNING`, `SUCCEEDED`, `FAILED` |
| Telemetry logs status and one joint | `telemetry_node.py`; [launch log](../ros2_ws/validation/policy_recovery/launch.log) |
| Configurable unsuccessful timeout | Separate wall-clock watchdog covers simulator startup and hangs; [timeout evidence](../ros2_ws/validation/20260919T231348Z-15524/integration.json) |
| Five policy episodes and defined success | [Selected evaluation](../artifacts/submission/local_stability/evaluation/summary.json), seeds 1001–1005; criteria in [RESULTS.md](../RESULTS.md) |
| Failures and limitations explained | [RESULTS.md](../RESULTS.md): initial 0/5, selected 5/5, inward/edge-supported bracing, clean-stance 0/5 and strict joint-limit compliance 0/5 |
| Fresh ROS build, launch, literal CLI request and telemetry | [ROS validation](ros_validation.md), original fresh-build CLI evidence plus later learned-policy `SUCCEEDED` probe |
| Working source reproduction | [Final reproduction record](../artifacts/validation/final_reproduction/README.md), tested independently of the development source path |
| Setup, training, evaluation and ROS commands | [README](../README.md), [START_HERE](../START_HERE.md), executable launchers and pinned `uv.lock` |
| Meaningful GitHub history | [Private repository](https://github.com/sushanthsujeerkumar/hrs-x2-recovery), initialized before model/trainer/integration milestones, with regular pushed commits |

The scripted controller is explicitly an untrained unsuccessful comparison. It is not substituted for learned-policy evidence. The training itself was not blocked. Docker was optional and is not included.
