#!/usr/bin/env python3
"""Two bounded physical checks of static targets; not learned ground recovery."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from validate_ik_bridges import episode
from x2_recovery.common import ModelInfo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="artifacts/validation/floor_transition/gravity_offset_bridges_repeat",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError("Choose a fresh output directory to preserve evidence")
    output.mkdir(parents=True, exist_ok=True)
    info = ModelInfo(physics_profile="guarded_v2")
    results = []
    for name, start in [("half_knee_region", "deep_crouch"), ("seat_feet_hands", "supine")]:
        result, states = episode(info, name, start, gravity_offsets=True)
        results.append(result)
        label = f"{start}_{name}"
        (output / f"{label}.json").write_text(json.dumps(result, indent=2) + "\n")
        np.savez_compressed(output / f"{label}_states.npz", qpos=states)
        print(json.dumps({key: result[key] for key in
                          ["name", "start", "standing_return_pass", "reason", "limits"]}),
              flush=True)
    (output / "summary.json").write_text(json.dumps({
        "scope": "Two physical checks of static gravity-offset targets; no learned recovery",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "results": results,
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
