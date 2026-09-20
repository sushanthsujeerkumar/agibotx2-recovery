#!/usr/bin/env python3
"""Bounded, imperfect clock-based DAgger-style crouch demonstration aggregation.

The scripted teacher supplies labels at the current episode time, even when the
learner has drifted off its reference trajectory. These labels are not a proven
state-feedback correction, and failed/truncated trajectories are retained as
explicitly labelled augmentation rather than reported as successful references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import mujoco
import numpy as np
import torch

from validate_crouch_stage import crouch_target, normalized_action
from x2_recovery.common import (
    CONTROL_DT, MODEL_PATH, ModelInfo, observation_numpy, sensor_forces,
    ground_forces, success_conditions,
)
from x2_recovery.physics import TrajectoryLimits, reset_crouch
from x2_recovery.stance import stance_metrics


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rollout(info, actor, seed, teacher_weight, seconds, transition):
    model = info.model
    data = mujoco.MjData(model)
    reset_crouch(info, data, seed)
    initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
    previous = np.zeros(model.nu, dtype=np.float32)
    monitor = TrajectoryLimits(info)
    monitor.observe(data)
    low = crouch_target(info)
    values = {key: [] for key in ("observations", "actions", "executed_actions", "next_observations",
                                  "times_s", "time_deltas_s", "next_state_limits_ok")}
    end_reason = "duration"
    clean_hold = max_clean_hold = 0.
    for _ in range(round(seconds / CONTROL_DT)):
        time_s = float(data.time)
        obs = observation_numpy(info, data, previous, sensor_forces(info, data))
        if not np.isfinite(obs).all():
            end_reason = "nonfinite_observation"
            break
        with torch.inference_mode():
            predicted = actor(torch.from_numpy(obs)[None]).squeeze(0).numpy()
        if predicted.shape != (model.nu,) or not np.isfinite(predicted).all():
            end_reason = "invalid_actor_action"
            break
        phase = min(time_s / transition, 1.)
        phase = phase * phase * (3. - 2. * phase)
        teacher_action = normalized_action(info, low * (1. - phase) + info.nominal * phase)
        # Action clipping is the deployed policy interface; state is never clipped.
        executed = (teacher_weight * teacher_action +
                    (1. - teacher_weight) * np.clip(predicted, -1., 1.)).astype(np.float32)
        target = info.targets(executed)
        for _ in range(info.substeps):
            data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], target)
            mujoco.mj_step(model, data)
            monitor.observe(data)
            if not monitor.ok:
                end_reason = "trajectory_limit_violation"
                break
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                end_reason = "nonfinite_state"
                break
            if data.qpos[2] < .3:
                end_reason = "pelvis_below_0.3m"
                break
        previous = executed.copy()
        next_obs = observation_numpy(info, data, previous, sensor_forces(info, data))
        if not np.isfinite(next_obs).all():
            end_reason = "nonfinite_next_observation"
            break
        values["observations"].append(obs)
        values["actions"].append(teacher_action)
        values["executed_actions"].append(executed)
        values["next_observations"].append(next_obs)
        values["times_s"].append(time_s)
        values["time_deltas_s"].append(float(data.time) - time_s)
        values["next_state_limits_ok"].append(monitor.ok)
        checks = success_conditions(info, data, ground_forces(info, data))
        stance = stance_metrics(info, data)
        clean = all(checks.values()) and stance["posture_ok"] and monitor.ok
        clean_hold = clean_hold + float(data.time) - time_s if clean else 0.
        max_clean_hold = max(max_clean_hold, clean_hold)
        if end_reason != "duration":
            break
    checks = success_conditions(info, data, ground_forces(info, data))
    stance = stance_metrics(info, data)
    complete = abs(float(data.time) - seconds) < 1e-6
    clean = all(checks.values()) and stance["posture_ok"] and monitor.ok
    row = {
        "seed": seed, "teacher_weight": teacher_weight,
        "policy_weight": 1. - teacher_weight, "samples": len(values["actions"]),
        "complete": complete, "truncated": not complete, "end_reason": end_reason,
        "sim_time_s": float(data.time), "terminal_pelvis_height_m": float(data.qpos[2]),
        "trajectory_limits": monitor.report(), "terminal_standing_checks": checks,
        "terminal_stance": stance, "terminal_clean": bool(clean),
        "max_clean_hold_s_control_sampled": max_clean_hold,
        "blended_rollout_clean_pass": bool(complete and monitor.ok and clean and max_clean_hold >= 2. - 1e-8),
        "label_warning": "Clock-based teacher action, not a verified corrective label for this off-reference state",
    }
    arrays = {key: np.asarray(value, dtype=(np.float32 if key in
              ("observations", "actions", "executed_actions", "next_observations") else
              bool if key == "next_state_limits_ok" else np.float64)) for key, value in values.items()}
    arrays.update(initial_qpos=initial_qpos, initial_qvel=initial_qvel)
    return row, arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/validation/crouch_stage/augmented"))
    parser.add_argument("--source", type=Path,
                        default=Path("artifacts/validation/crouch_stage/reference/training_pairs.npz"))
    parser.add_argument("--teacher-weight", type=float, nargs="+", default=[.9, .6])
    parser.add_argument("--first-seed", type=int, default=121)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--transition-seconds", type=float, default=3.)
    args = parser.parse_args()
    if args.episodes < 1 or args.seconds <= 0. or args.transition_seconds <= 0.:
        parser.error("Counts and durations must be positive")
    if any(not 0. <= weight <= 1. for weight in args.teacher_weight):
        parser.error("Teacher weights must be within [0,1]")
    if len(set(args.teacher_weight)) != len(args.teacher_weight):
        parser.error("Teacher weights must be distinct")
    if not np.isclose(args.seconds / CONTROL_DT, round(args.seconds / CONTROL_DT)):
        parser.error("Episode duration must be a control-period multiple")
    seeds = list(range(args.first_seed, args.first_seed + args.episodes))
    if min(seeds) < 0 or max(seeds) >= 3001:
        parser.error("Use training seeds below 3001; independent physical-test seeds are reserved")
    args.output.mkdir(parents=True, exist_ok=True)
    destination = args.output / "training_pairs.npz"
    if destination.exists():
        parser.error("Choose a fresh output directory; an aggregated dataset already exists")
    torch.set_num_threads(2)
    actor = torch.jit.load(str(args.actor), map_location="cpu").eval()
    info = ModelInfo(physics_profile="guarded_v2")
    root = Path(__file__).resolve().parents[1]
    source = np.load(args.source, allow_pickle=False)
    if set(source["seeds"]) & set(range(3001, 3006)):
        parser.error("Source includes reserved physical-test seeds")
    actor_digest = sha256(args.actor)
    files = [Path(__file__), Path(__file__).with_name("validate_crouch_stage.py"), MODEL_PATH,
             MODEL_PATH.parent / "model_metadata.json"]
    files += [root / f"src/x2_recovery/{name}.py" for name in ("common", "physics", "stance")]
    config = {
        "method": "imperfect_clock_based_dagger_style_augmentation",
        "warning": "The teacher is an unchanged time-indexed trajectory, not a state-feedback expert; labels may be poor after drift",
        "actor": str(args.actor), "actor_sha256": actor_digest,
        "source_dataset": str(args.source), "source_dataset_sha256": sha256(args.source),
        "training_seeds": seeds, "teacher_weights": args.teacher_weight,
        "episode_seconds": args.seconds, "transition_seconds": args.transition_seconds,
        "physics_profile": "guarded_v2", "physics_dt_s": float(info.model.opt.timestep),
        "control_dt_s": CONTROL_DT, "previous_action": "previous executed blended action; zero at reset",
        "label_action": "unchanged teacher's normalized action at current simulation time",
        "stopping": "first per-physics-step limit violation, nonfinite state, pelvis below 0.3 m, or duration",
        "retention": "Retain finite pre-action observations/teacher labels including final finite truncated transition; flag next-state compliance",
        "validation": "All seeds in this dataset are development seeds; no independent physical-test seeds are used",
        "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_sha256": {str(path.relative_to(root)): sha256(path) for path in files},
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    base_n, base_e = len(source["actions"]), len(source["seeds"])
    column_names = ("observations", "actions", "next_observations", "times_s",
                    "executed_actions", "time_deltas_s", "next_state_limits_ok")
    columns = {
        key: [source[key] if key in source.files else source["actions"] if key == "executed_actions" else
              np.full(base_n, CONTROL_DT) if key == "time_deltas_s" else np.ones(base_n, dtype=bool)]
        for key in column_names
    }
    initial_qpos, initial_qvel = [source["initial_qpos"]], [source["initial_qvel"]]
    episode_seeds = source["seeds"].tolist()
    lengths = np.diff(source["episode_offsets"]).tolist()
    teacher_weights = source["teacher_weights"].tolist() if "teacher_weights" in source.files else [1.] * base_e
    kinds = source["source_kind"].tolist() if "source_kind" in source.files else ["scripted_reference"] * base_e
    end_reasons = source["episode_end_reasons"].tolist() if "episode_end_reasons" in source.files else ["duration"] * base_e
    actor_hashes = source["actor_sha256"].tolist() if "actor_sha256" in source.files else [""] * base_e
    rows = []
    for weight in args.teacher_weight:
        for seed in seeds:
            row, arrays = rollout(info, actor, seed, weight, args.seconds, args.transition_seconds)
            rows.append(row)
            (args.output / f"weight_{weight:g}_seed_{seed}.json").write_text(json.dumps(row, indent=2) + "\n")
            if row["samples"]:
                for key in column_names:
                    columns[key].append(arrays[key])
                initial_qpos.append(arrays["initial_qpos"][None])
                initial_qvel.append(arrays["initial_qvel"][None])
                episode_seeds.append(seed)
                lengths.append(row["samples"])
                teacher_weights.append(weight)
                kinds.append("clock_labelled_blended_rollout")
                end_reasons.append(row["end_reason"])
                actor_hashes.append(actor_digest)
            print(json.dumps({key: row[key] for key in ("seed", "teacher_weight", "samples", "sim_time_s",
                              "end_reason", "terminal_clean", "trajectory_limits")}), flush=True)
    dataset = {key: np.concatenate(arrays, axis=0) for key, arrays in columns.items()}
    dataset.update(
        initial_qpos=np.concatenate(initial_qpos, axis=0), initial_qvel=np.concatenate(initial_qvel, axis=0),
        seeds=np.asarray(episode_seeds, dtype=np.int64),
        episode_offsets=np.cumsum([0] + lengths, dtype=np.int64),
        episode_ids=np.repeat(np.arange(len(lengths), dtype=np.int32), lengths),
        joint_names=source["joint_names"], teacher_weights=np.asarray(teacher_weights),
        source_kind=np.asarray(kinds), episode_end_reasons=np.asarray(end_reasons),
        actor_sha256=np.asarray(actor_hashes),
    )
    for key in ("observations", "actions", "executed_actions", "next_observations"):
        assert np.isfinite(dataset[key]).all(), key
    for start, end in zip(dataset["episode_offsets"][:-1], dataset["episode_offsets"][1:]):
        assert not np.any(dataset["observations"][start, -31:])
        assert np.array_equal(dataset["next_observations"][start:end, -31:], dataset["executed_actions"][start:end])
        assert np.array_equal(dataset["observations"][start+1:end], dataset["next_observations"][start:end-1])
    np.savez_compressed(destination, **dataset)
    summary = {
        "method": config["method"], "label_warning": config["warning"],
        "source_samples": base_n, "source_episodes": base_e,
        "augmented_samples": len(dataset["actions"]) - base_n,
        "augmented_episodes_attempted": len(rows), "augmented_episodes_captured": len(lengths) - base_e,
        "complete_blended_rollouts": sum(row["complete"] for row in rows),
        "clean_blended_rollout_passes": sum(row["blended_rollout_clean_pass"] for row in rows),
        "trajectory_limit_passes": sum(row["trajectory_limits"]["ok"] for row in rows),
        "combined_samples": len(dataset["actions"]), "combined_episodes": len(lengths),
        "dataset_sha256": sha256(destination),
        "schema_note": "Same observations/actions/seeds[episode_ids] interface as reference; actions are teacher labels, executed_actions drive dynamics. Per-episode teacher_weights/source_kind/episode_end_reasons/actor_sha256 explain provenance.",
        "split_warning": "Stratify development validation by source seed; holding out all four new seeds removes all augmentation from fitting",
        "checks": {"all_stored_values_finite": True, "previous_actions_match_executed_actions": True,
                   "adjacent_observations_match": True, "episode_ids_remapped": True,
                   "independent_physical_test_seeds_absent": True},
        "results": rows,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}), flush=True)


if __name__ == "__main__":
    main()
