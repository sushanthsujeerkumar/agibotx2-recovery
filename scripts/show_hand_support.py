#!/usr/bin/env python3
"""Show the 16 s scripted hand-support reference, not a policy or floor recovery."""
import json
import time

import numpy as np

from screen_supported_lowering import target
from x2_recovery.physics import deep_crouch_target
from x2_recovery.runtime import RecoveryRuntime


def main():
    runtime = RecoveryRuntime(
        render=True, assess_stance=True, physics_profile="guarded_v2",
        reset_mode="deep_crouch", seed=9101,
    )
    runtime.display_label = "SCRIPTED SUPPORT REFERENCE - NOT PPO OR FLOOR RECOVERY"
    deep = deep_crouch_target(runtime.info)
    hands = target(runtime.info, 2.32, 0.9)
    times = [0.0, 4.0, 6.0, 10.0, 13.0, 16.0]
    poses = [deep, hands, hands, deep, runtime.info.nominal, runtime.info.nominal]

    def desired_target():
        t = float(runtime.data.time)
        index = min(max(np.searchsorted(times, t, side="right") - 1, 0), len(times) - 2)
        fraction = np.clip((t - times[index]) / (times[index + 1] - times[index]), 0, 1)
        fraction = fraction * fraction * (3 - 2 * fraction)
        return poses[index] * (1 - fraction) + poses[index + 1] * fraction

    # Viewer-only script hook; this does not load or modify any PPO actor.
    runtime._scripted_target = desired_target
    try:
        for _ in range(800):
            started = time.monotonic()
            status = runtime.step()
            if (not runtime.viewer.is_running() or status["invalid"]
                    or not status["trajectory_limits"]["ok"]):
                break
            time.sleep(max(0.0, 0.02 - (time.monotonic() - started)))
        print(json.dumps({key: value for key, value in status.items()
                          if key not in ["joint_names", "joint_positions"]}))
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
