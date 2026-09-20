# Running the submitted model

From this folder:

```bash
./RUN_DEMO.sh       # visible recovery
./EVALUATE.sh       # five headless evaluation episodes
./RUN_RECOVERY.sh render:=true  # launch both ROS nodes
```

On a fresh machine, first run `bash scripts/setup.sh`; ROS needs Ubuntu 24.04 / ROS 2 Jazzy. See the [README](README.md) for complete setup and the service call.

The selected model is `artifacts/submission/final_recovery/actor.pt`. It combines a project-generated frozen motion prior with PPO feedback and passed the five-episode assessment. [Results](RESULTS.md) describe the checks and limitations. Older reference/vendor launchers reproduce earlier experiments and are not the submission entry points.
