#!/usr/bin/env python3
"""Bounded scripted deeper-crouch feasibility screen, not learned recovery."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from x2_recovery.common import (
    CONTROL_DT, MODEL_PATH, ModelInfo, ground_forces, success_conditions,
)
from x2_recovery.physics import TrajectoryLimits, reset_balance
from x2_recovery.stance import stance_metrics


CANDIDATES = [
    (1.6, -.70, 0.), (1.7, -.70, 0.), (1.8, -.70, 0.), (1.9, -.70, 0.),
    (1.7, -.65, 0.), (1.8, -.72, 0.), (1.9, -.72, .12), (1.9, -.72, -.12),
]
LOWER_SECONDS = 4.
BOTTOM_HOLD_SECONDS = 1.
RETURN_SECONDS = 4.
FINAL_HOLD_SECONDS = 4.
RETURN_END = LOWER_SECONDS + BOTTOM_HOLD_SECONDS + RETURN_SECONDS
DURATION = RETURN_END + FINAL_HOLD_SECONDS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pose(info, knee, ankle, waist):
    target = info.nominal.copy()
    for side in ("left", "right"):
        for part, value in (("hip_pitch", -(knee + ankle)), ("knee", knee), ("ankle_pitch", ankle)):
            target[info.names.index(f"{side}_{part}_joint")] = value
    target[info.names.index("waist_pitch_joint")] = waist
    return target


def target_checks(info, target):
    lower_error = np.maximum(info.lower - target, 0.)
    upper_error = np.maximum(target - info.upper, 0.)
    guarded_error = np.maximum.reduce((info.lower + .05 - target,
                                      target - info.upper + .05, np.zeros_like(target)))
    return {
        "inside_published_bounds": bool(max(lower_error.max(), upper_error.max()) <= 1e-10),
        "inside_guarded_command_margin": bool(guarded_error.max() <= 1e-10),
        "maximum_guarded_target_clamp_rad": float(guarded_error.max()),
        "minimum_distance_to_guarded_boundary_rad": float(np.minimum(target - info.lower - .05,
                                                                     info.upper - .05 - target).min()),
    }


def snapshot(info, data, forces):
    return {
        "time_s": float(data.time), "pelvis_height_m": float(data.qpos[2]),
        "joint_positions_rad": data.qpos[info.qadr].tolist(),
        "exact_ground_forces_n": forces.tolist(),
        "stance": stance_metrics(info, data),
        "standing_checks": success_conditions(info, data, forces),
        "linear_speed_m_s": float(np.linalg.norm(data.qvel[:3])),
        "angular_speed_rad_s": float(np.linalg.norm(data.qvel[3:6])),
    }


def episode(info, candidate, seed):
    knee, ankle, waist = candidate
    target = pose(info, *candidate)
    bound_check = target_checks(info, target)
    if not bound_check["inside_guarded_command_margin"]:
        raise ValueError(f"Screen candidate would be clipped: {bound_check}")
    model = info.model
    data = mujoco.MjData(model)
    reset_balance(info, data, seed)
    monitor = TrajectoryLimits(info)
    monitor.observe(data)
    forces = ground_forces(info, data)
    initial = snapshot(info, data, forces)
    minimum = initial
    bottom = None
    hold = max_hold = 0.
    min_foot_forces = forces[:2].copy()
    max_other_support = float(forces[2])
    max_foot_bracing = 0.
    max_command_clamp = 0.
    reason = "duration"
    clean = False
    for control_step in range(round(DURATION / CONTROL_DT)):
        time_s = control_step * CONTROL_DT
        if time_s < LOWER_SECONDS:
            phase = time_s / LOWER_SECONDS
        elif time_s < LOWER_SECONDS + BOTTOM_HOLD_SECONDS:
            phase = 1.
        else:
            phase = max(0., 1. - (time_s - LOWER_SECONDS - BOTTOM_HOLD_SECONDS) / RETURN_SECONDS)
        phase = phase * phase * (3. - 2. * phase)
        command = info.nominal * (1. - phase) + target * phase
        clamped = np.clip(command, info.lower + .05, info.upper - .05)
        max_command_clamp = max(max_command_clamp, float(np.abs(command - clamped).max()))
        for _ in range(info.substeps):
            data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], command)
            mujoco.mj_step(model, data)
            monitor.observe(data)
            forces = ground_forces(info, data)
            stance = stance_metrics(info, data)
            checks = success_conditions(info, data, forces)
            min_foot_forces = np.minimum(min_foot_forces, forces[:2])
            max_other_support = max(max_other_support, float(forces[2]))
            max_foot_bracing = max(max_foot_bracing, stance["foot_bracing_force_n"])
            if data.qpos[2] < minimum["pelvis_height_m"]:
                minimum = snapshot(info, data, forces)
            if bottom is None and data.time >= LOWER_SECONDS + BOTTOM_HOLD_SECONDS - 1e-8:
                bottom = snapshot(info, data, forces)
            clean = all(checks.values()) and stance["posture_ok"] and monitor.ok
            if data.time >= RETURN_END - 1e-8:
                hold = hold + model.opt.timestep if clean else 0.
                max_hold = max(max_hold, hold)
            if not monitor.ok:
                reason = "trajectory_limit_violation"
                break
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                reason = "nonfinite_state"
                break
            if data.qpos[2] < .3:
                reason = "pelvis_below_0.3m"
                break
        if reason != "duration":
            break
    complete = abs(float(data.time) - DURATION) < 1e-6
    passed = complete and monitor.ok and bool(clean) and max_hold >= 2. - 1e-8 and max_command_clamp <= 1e-10
    return {
        "candidate": {"hip_pitch_rad": -(knee + ankle), "knee_rad": knee,
                      "ankle_pitch_rad": ankle, "waist_pitch_rad": waist},
        "seed": seed, "scripted_reference_pass": passed,
        "complete": complete, "end_reason": reason, "sim_time_s": float(data.time),
        "target_bounds": bound_check, "maximum_command_clamp_rad": max_command_clamp,
        "initial": initial, "minimum_height_state": minimum, "bottom_hold_end": bottom,
        "final": snapshot(info, data, forces),
        "clean_hold_after_return_s": max_hold, "terminal_clean_hold_s": hold,
        "minimum_exact_foot_ground_forces_n": min_foot_forces.tolist(),
        "maximum_exact_nonfoot_ground_force_n": max_other_support,
        "maximum_foot_bracing_force_n": max_foot_bracing,
        "trajectory_limits": monitor.report(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/validation/crouch_stage/deeper_reference"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        parser.error("Choose a fresh output directory")
    info = ModelInfo(physics_profile="guarded_v2")
    if not target_checks(info, info.nominal)["inside_guarded_command_margin"]:
        raise ValueError("Nominal pose lies outside guarded command margins")
    root = Path(__file__).resolve().parents[1]
    files = [Path(__file__), MODEL_PATH, MODEL_PATH.parent / "model_metadata.json"]
    files += [root / f"src/x2_recovery/{name}.py" for name in ("common", "physics", "stance")]
    config = {
        "task": "Scripted deeper-crouch reference feasibility only; NOT learned or ground recovery",
        "physics_profile": "guarded_v2", "physics_dt_s": float(info.model.opt.timestep),
        "control_dt_s": CONTROL_DT, "screen_seed": 4001, "validation_seeds": list(range(5001, 5006)),
        "candidate_count_cap": 8, "candidates": [dict(knee_rad=k, ankle_pitch_rad=a,
                                                      hip_pitch_rad=-(k+a), waist_pitch_rad=w) for k,a,w in CANDIDATES],
        "lower_seconds": LOWER_SECONDS, "bottom_hold_seconds": BOTTOM_HOLD_SECONDS,
        "return_seconds": RETURN_SECONDS, "final_hold_seconds": FINAL_HOLD_SECONDS,
        "interpolation": "smoothstep", "reset": "physically settled reset_balance; no later state edits",
        "checks": "Exact ground/foot-bracing contacts, clean standing and latched trajectory limits at every physics step",
        "selection": "Deepest minimum pelvis height among screen passes; selected before five additional validation seeds",
        "limits": {name: {"lower_rad": float(info.lower[j]), "upper_rad": float(info.upper[j]),
                          "guarded_lower_rad": float(info.lower[j]+.05), "guarded_upper_rad": float(info.upper[j]-.05)}
                   for j, name in enumerate(info.names)},
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {str(path.relative_to(root)): digest(path) for path in files},
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    screen = []
    for index, candidate in enumerate(CANDIDATES):
        row = episode(info, candidate, 4001)
        row["candidate_index"] = index
        screen.append(row)
        (args.output / f"screen_{index:02d}.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps({"split": "screen", "candidate": row["candidate"], "passed": row["scripted_reference_pass"],
                          "minimum_height_m": row["minimum_height_state"]["pelvis_height_m"],
                          "end_reason": row["end_reason"], "clean_hold_s": row["clean_hold_after_return_s"],
                          "limits": row["trajectory_limits"]}), flush=True)
    passing = [row for row in screen if row["scripted_reference_pass"]]
    best = min(passing, key=lambda row: row["minimum_height_state"]["pelvis_height_m"]) if passing else None
    validation = []
    if best is not None:
        candidate = CANDIDATES[best["candidate_index"]]
        for seed in range(5001, 5006):
            row = episode(info, candidate, seed)
            validation.append(row)
            (args.output / f"validation_{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
            print(json.dumps({"split": "validation", "seed": seed, "passed": row["scripted_reference_pass"],
                              "minimum_height_m": row["minimum_height_state"]["pelvis_height_m"],
                              "clean_hold_s": row["clean_hold_after_return_s"], "limits": row["trajectory_limits"]}), flush=True)
    summary = {
        "task": config["task"], "screened_candidates": len(screen), "screen_passes": len(passing),
        "selected_candidate_index": best["candidate_index"] if best else None,
        "selected_candidate": best["candidate"] if best else None,
        "validation_episodes": len(validation),
        "validation_passes": sum(row["scripted_reference_pass"] for row in validation),
        "validation_limit_passes": sum(row["trajectory_limits"]["ok"] for row in validation),
        "screen_results": screen, "validation_results": validation,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key,value in summary.items() if not key.endswith("results")}), flush=True)


if __name__ == "__main__":
    main()
