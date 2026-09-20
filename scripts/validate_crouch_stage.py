#!/usr/bin/env python3
"""Capture a checked scripted crouch-to-standing teacher, not RL recovery.

The teacher begins at reset_crouch's physically attained state. No position or
velocity state is clipped, and no contacts or model collision flags are changed.
Only valid complete training episodes enter the observation/action dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np

from x2_recovery.common import (
    CONTROL_DT, HOLD_SECONDS, MODEL_PATH, ModelInfo, ground_forces,
    observation_numpy, sensor_forces, success_conditions,
)
from x2_recovery.physics import TrajectoryLimits, reset_crouch
from x2_recovery.stance import stance_metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crouch_target(info):
    target = info.nominal.copy()
    for side in ("left", "right"):
        for joint, value in (("hip_pitch", -.7), ("knee", 1.4), ("ankle_pitch", -.7)):
            target[info.names.index(f"{side}_{joint}_joint")] = value
    return target


def normalized_action(info, target):
    delta = target - info.nominal
    scale = np.where(delta >= 0., info.action_positive, info.action_negative)
    action = (delta / scale).astype(np.float32)
    if np.any(np.abs(action) > 1.) or not np.allclose(info.targets(action), target, atol=1e-7):
        raise ValueError("Teacher target does not round-trip through full-range actions")
    return action


def episode(info, seed, duration, transition):
    model = info.model
    data = mujoco.MjData(model)
    reset_crouch(info, data, seed)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    initial_height = float(data.qpos[2])
    initial_checks = success_conditions(info, data, ground_forces(info, data))
    monitor = TrajectoryLimits(info)
    monitor.observe(data)
    low = crouch_target(info)
    previous = np.zeros(model.nu, dtype=np.float32)
    observations, actions, next_observations, times = [], [], [], []
    hold = max_hold = 0.
    first_hold = None
    clean = False
    min_height = initial_height
    max_bracing = 0.
    last_forces = ground_forces(info, data)
    checks, stance = initial_checks, stance_metrics(info, data)
    for step in range(round(duration / CONTROL_DT)):
        time_s = step * CONTROL_DT
        phase = min(time_s / transition, 1.)
        phase = phase * phase * (3. - 2. * phase)
        target = low * (1. - phase) + info.nominal * phase
        action = normalized_action(info, target)
        # Use exactly the float32 action represented in the saved dataset.
        target = info.targets(action)
        observations.append(observation_numpy(info, data, previous, sensor_forces(info, data)))
        actions.append(action)
        times.append(float(data.time))
        for _ in range(info.substeps):
            data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], target)
            mujoco.mj_step(model, data)
            monitor.observe(data)
            last_forces = ground_forces(info, data)
            checks = success_conditions(info, data, last_forces)
            stance = stance_metrics(info, data)
            clean = all(checks.values()) and stance["posture_ok"] and monitor.ok
            hold = hold + model.opt.timestep if clean else 0.
            max_hold = max(max_hold, hold)
            if first_hold is None and hold >= HOLD_SECONDS - 1e-8:
                first_hold = float(data.time)
            min_height = min(min_height, float(data.qpos[2]))
            max_bracing = max(max_bracing, stance["foot_bracing_force_n"])
        previous = action.copy()
        next_observations.append(observation_numpy(info, data, previous, sensor_forces(info, data)))
        if not monitor.ok or not np.isfinite(data.qpos).all() or data.qpos[2] < .3:
            break
    complete = abs(float(data.time) - duration) < 1e-6
    passed = complete and first_hold is not None and bool(clean) and monitor.ok
    row = {
        "seed": seed, "scripted_reference_pass": passed,
        "complete": complete, "sim_time_s": float(data.time),
        "initial_pelvis_height_m": initial_height, "initial_standing_checks": initial_checks,
        "minimum_pelvis_height_m": min_height, "final_pelvis_height_m": float(data.qpos[2]),
        "first_clean_hold_s": first_hold, "max_clean_hold_s": max_hold,
        "terminal_clean_hold_s": hold, "terminal_clean_standing": bool(clean),
        "final_base_linear_speed_m_s": float(np.linalg.norm(data.qvel[:3])),
        "final_base_angular_speed_rad_s": float(np.linalg.norm(data.qvel[3:6])),
        "maximum_foot_bracing_force_n": max_bracing,
        "final_exact_ground_forces_n": last_forces.tolist(),
        "final_standing_checks": checks, "final_stance": stance,
        "trajectory_limits": monitor.report(), "control_samples": len(actions),
    }
    arrays = {
        "observations": np.asarray(observations, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "next_observations": np.asarray(next_observations, dtype=np.float32),
        "times_s": np.asarray(times, dtype=np.float64),
        "initial_qpos": initial_qpos, "initial_qvel": initial_qvel,
    }
    return row, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/validation/crouch_stage/reference"))
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--transition-seconds", type=float, default=3.)
    parser.add_argument("--validation-first-seed", type=int, default=3001)
    parser.add_argument("--validation-episodes", type=int, default=5)
    parser.add_argument("--training-first-seed", type=int, default=101)
    parser.add_argument("--training-episodes", type=int, default=16)
    args = parser.parse_args()
    if args.seconds <= 0 or args.transition_seconds <= 0 or args.seconds < args.transition_seconds + HOLD_SECONDS:
        parser.error("Duration must include positive transition plus the clean standing hold")
    if not np.isclose(args.seconds / CONTROL_DT, round(args.seconds / CONTROL_DT)):
        parser.error("Duration must be a multiple of the control period")
    if min(args.validation_episodes, args.training_episodes) < 1:
        parser.error("Both seed sets must be non-empty")
    validation_seeds = list(range(args.validation_first_seed, args.validation_first_seed + args.validation_episodes))
    training_seeds = list(range(args.training_first_seed, args.training_first_seed + args.training_episodes))
    if set(validation_seeds) & set(training_seeds):
        parser.error("Training and validation seed sets must be disjoint")
    args.output.mkdir(parents=True, exist_ok=True)
    info = ModelInfo(physics_profile="guarded_v2")
    root = Path(__file__).resolve().parents[1]
    source_paths = [Path(__file__), MODEL_PATH, MODEL_PATH.parent / "model_metadata.json"]
    source_paths += [root / f"src/x2_recovery/{name}.py" for name in ("common", "physics", "stance")]
    config = {
        "task": "Scripted moderate crouch-to-standing teacher; NOT learned RL or supine recovery",
        "physics_profile": "guarded_v2", "physics_dt_s": float(info.model.opt.timestep),
        "control_dt_s": CONTROL_DT, "episode_duration_s": args.seconds,
        "transition_duration_s": args.transition_seconds, "interpolation": "smoothstep s*s*(3-2*s)",
        "reset": "reset_crouch: physically generated moderate crouch, never state-clipped",
        "reset_time_excluded_from_episode_s": 3.9,
        "teacher_crouch_command_rad": {"hip_pitch": -.7, "knee": 1.4, "ankle_pitch": -.7},
        "other_joints": "nominal", "joint_names": info.names,
        "initial_previous_action": "31 zeros, matching deployed runtime",
        "observation_contacts": "sensor_forces, matching deployed policy inputs",
        "validation_contacts": "exact solver ground forces and foot-to-foot forces",
        "checks_frequency": "Every physics step for limits, standing and stance; reset routines also audit limits",
        "pass_rule": "Complete 10 s (or configured duration), >=2 s clean hold, clean terminal stance, no limit violations",
        "validation_seeds": validation_seeds, "training_seeds": training_seeds,
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in source_paths},
        "software": {"numpy": np.__version__, "mujoco": mujoco.__version__},
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = {"validation": [], "training": []}
    valid_data, valid_seeds = [], []
    for split, seeds in (("validation", validation_seeds), ("training", training_seeds)):
        for seed in seeds:
            row, arrays = episode(info, seed, args.seconds, args.transition_seconds)
            results[split].append(row)
            (args.output / f"{split}_{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
            if split == "training" and row["scripted_reference_pass"]:
                valid_data.append(arrays)
                valid_seeds.append(seed)
            print(json.dumps({"split": split, "seed": seed, "passed": row["scripted_reference_pass"],
                              "clean_hold_s": row["max_clean_hold_s"], "limits": row["trajectory_limits"]}), flush=True)
    dataset_path = args.output / "training_pairs.npz"
    if valid_data:
        lengths = [len(arrays["actions"]) for arrays in valid_data]
        dataset = {key: np.concatenate([arrays[key] for arrays in valid_data], axis=0)
                   for key in ("observations", "actions", "next_observations", "times_s")}
        dataset.update(
            episode_ids=np.repeat(np.arange(len(valid_data), dtype=np.int32), lengths),
            episode_offsets=np.cumsum([0] + lengths, dtype=np.int64),
            seeds=np.asarray(valid_seeds, dtype=np.int64),
            initial_qpos=np.stack([arrays["initial_qpos"] for arrays in valid_data]),
            initial_qvel=np.stack([arrays["initial_qvel"] for arrays in valid_data]),
            joint_names=np.asarray(info.names),
        )
        for key in ("observations", "actions", "next_observations"):
            assert np.isfinite(dataset[key]).all(), f"Nonfinite dataset {key}"
        assert dataset["observations"].shape[1] == 106 and dataset["actions"].shape[1] == 31
        # Last 31 observation elements contain the previous commanded action.
        for start, end in zip(dataset["episode_offsets"][:-1], dataset["episode_offsets"][1:]):
            assert not np.any(dataset["observations"][start, -31:])
            assert np.array_equal(dataset["observations"][start + 1:end, -31:], dataset["actions"][start:end - 1])
        np.savez_compressed(dataset_path, **dataset)
    elif dataset_path.exists():
        dataset_path.unlink()
    summary = {
        "task": config["task"], "physics_profile": "guarded_v2",
        "validation_episodes": len(results["validation"]),
        "validation_clean_passes": sum(r["scripted_reference_pass"] for r in results["validation"]),
        "validation_limit_passes": sum(r["trajectory_limits"]["ok"] for r in results["validation"]),
        "training_episodes_attempted": len(results["training"]),
        "training_episodes_captured": len(valid_data), "captured_training_seeds": valid_seeds,
        "rejected_training_seeds": [r["seed"] for r in results["training"] if not r["scripted_reference_pass"]],
        "training_samples": sum(len(arrays["actions"]) for arrays in valid_data),
        "dataset_sha256": sha256(dataset_path) if valid_data else None,
        "dataset_schema": {
            "observations": "float32 [N,106], pre-action observation, with sensor touch features",
            "actions": "float32 [N,31], full-range normalized position-target action, held 20 ms",
            "next_observations": "float32 [N,106], post-action observation with current action as previous action",
            "times_s": "float64 [N], pre-action simulation time, resetting to zero per episode",
            "episode_ids": "int32 [N], index into seeds/initial states",
            "episode_offsets": "int64 [E+1], half-open boundaries into sample arrays",
            "seeds": "int64 [E], valid TRAINING seeds only",
            "initial_qpos": "float64 [E,38], physically attained reset positions",
            "initial_qvel": "float64 [E,37], physically attained reset velocities",
            "joint_names": "unicode [31], action/actuator ordering",
        },
        "validation_results": results["validation"], "training_results": results["training"],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("validation_results", "training_results", "dataset_schema")}), flush=True)


if __name__ == "__main__":
    main()
