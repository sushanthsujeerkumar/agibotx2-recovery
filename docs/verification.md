# Release verification

The selected model is frozen at PPO update 1,548. Verification did not replace it with the short resume-test output.

| Check | Outcome |
|---|---|
| Five new assessment seeds, 30001–30005 | 5/5 full 15-second recoveries |
| Earlier final-checkpoint seeds, 27201–27210 | 10/10 recoveries |
| Python regression suite | 23 passed; TorchScript deprecation warnings only |
| ROS package tests | 28 passed, no failures/errors |
| Fresh ROS build and real service tests | Success, busy rejection, telemetry and both timeout paths passed |
| Separate source-folder evaluation before cleanup | Same five assessment seeds passed 5/5 |
| Checkpoint resume from separate folder | Updates 1,548 → 1,550; 12,288 added transitions; saved and stopped |
| Re-export from full checkpoint | Exact action equality on 1,000 observations against submitted actor |
| Regenerate motion prior | All eight reference examples succeeded; fitted actions matched exactly; 10/10 validation |
| Shell/Python syntax and current documentation links | Passed |
| Frozen model hashes | Verified against `manifest.json` |

Before the extended-demo update, the remaining 16 Python tests, 28 ROS tests, five assessment episodes and fresh ROS integration checks passed again. Seven Python tests and two ROS cases belonged only to removed experimental controllers. Their earlier test records remain in Git history.

The earlier separate folder test explicitly checked that `x2_recovery.common.ROOT` resolved to that folder. It reused the installed locked Python dependencies and had no access requirement for development experiment paths. A separate attempt to download a new Python environment was stopped after repeated large CUDA download restarts; a fully clean Python installation was not completed during this release check. This does not affect the fresh colcon builds or the recorded policy results. The setup script and lockfile are supplied, and a new machine needs network access for those dependencies.

Raw verification records are in [`artifacts/submission/final_recovery/verification/`](../artifacts/submission/final_recovery/verification/). The primary ROS CLI logs are in the adjacent `ros/` directory. The source archive includes the cleaned submission files and the reviewed extended-demo changes, excluding virtual environments, caches and unfinished local experiments. Its checksum, base Git revision and uncommitted changes are recorded alongside the archive in `HRS_X2_Release.json`; the separate `HRS_X2_History.bundle` preserves the earlier committed history. No new commit or push was made for the demo update.

## Extended interactive demo

`RUN_DEMO.sh` continues physical control after the 15-second assessment. R restarts the same seeded recovery, and closing the viewer ends the process. `RUN_DEMO.sh --once` retains the bounded demo. Seven new lifecycle tests cover continued stepping, early close, restart, fault pausing/restart, bounded mode and invalid headless usage. A real viewer smoke check completed recovery, dispatched the R callback, completed recovery again and closed cleanly; its result is in `verification/extended_demo_window.json`. The trained weights and the assessment/ROS time limits are unchanged.
