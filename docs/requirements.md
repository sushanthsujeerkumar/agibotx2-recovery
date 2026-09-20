# Assessment requirements and evidence

This maps the official HRS task to the submitted version. The final policy is reference-guided PPO; its fixed motion prior is described explicitly rather than attributed to reinforcement learning.

| Task requirement | Implementation / evidence |
|---|---|
| X2 URDF, simulator and rationale | [Model](model.md), source URDF and license in `assets/x2`; MuJoCo with mjlab/Warp training |
| Floating base, floor and collision model | `assets/x2/scene.xml`, metadata and model validation |
| Supine start without initial intersection | `reset_cpu` in `common.py`; settled reset protocol in [environment](environment.md) |
| Joint and actuator limits | Guarded torque servo; every-step audit; no violations in the five final assessment trajectories |
| Observations, actions, reward and termination | [Environment specification](environment.md), `env.py`, `full_recovery_env.py` |
| Run RL, save checkpoint and reward plot | [Training](training.md); final `checkpoint.pt`, `actor.pt`, `progress.jsonl`, `training_curve.png` |
| Two nodes, fresh colcon build, one launch | `x2_recovery_ros`, `recovery.launch.py`, [ROS validation](ros_validation.md) |
| Accept before execution; reject while busy | Service queue and subprocess supervisor; `ros/success.json`, `ros/cli_busy.log` |
| Status, actual joints and timeout failure | Required topics, telemetry node and independent watchdog; `ros/launch.log`, timeout JSON files |
| Five episodes, clear success rule and results | [Results](../RESULTS.md), `evaluation_5.json`, complete 15-second episodes |
| Fresh launch and literal CLI test evidence | `scripts/validate_final_ros.sh`, saved `ros/cli_*.log` |
| Dependencies, hardware, reproduction and limitations | [README](../README.md), [training](training.md), `uv.lock` |
| Development history in GitHub | Existing repository and chronological commits; final local changes prepared without rewriting earlier history |

Evidence paths in the table are relative to `artifacts/submission/final_recovery/` unless otherwise shown. Earlier failed experiments remain available for inspection. Repository sharing/publishing is a separate handoff step.
