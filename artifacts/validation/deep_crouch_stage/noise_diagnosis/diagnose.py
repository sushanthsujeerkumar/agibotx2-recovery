"""One paired, at-most-one-second CPU normalization/noise diagnostic."""
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from x2_recovery.common import ModelInfo, observation_numpy, sensor_forces
from x2_recovery.physics import reset_deep_crouch, TrajectoryLimits


OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[3]
ACTOR = ROOT / "artifacts/experiments/crouch_ppo/actor.pt"
CHECKPOINT = ACTOR.with_name("checkpoint.pt")
GROUPS = {"joint_position": (0, 31), "joint_velocity": (31, 62), "gravity": (62, 65),
          "base_linear_velocity": (65, 68), "base_angular_velocity": (68, 71),
          "pelvis_height": (71, 72), "touch_flags": (72, 75), "previous_actions": (75, 106)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    info = ModelInfo(physics_profile="guarded_v2")
    actor = torch.jit.load(str(ACTOR), map_location="cpu").eval()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    buffers = dict(actor.named_buffers())
    for key,value in buffers.items():
        assert torch.equal(value, checkpoint["actor_state_dict"][key]), key
    mean = buffers["obs_normalizer._mean"].numpy().reshape(-1)
    std = buffers["obs_normalizer._std"].numpy().reshape(-1)
    eps = float(actor.obs_normalizer.eps)
    names = (["joint_position:" + name for name in info.names] +
             ["joint_velocity_scaled:" + name for name in info.names] +
             ["gravity:" + axis for axis in "xyz"] +
             ["base_linear_velocity:" + axis for axis in "xyz"] +
             ["base_angular_velocity_scaled:" + axis for axis in "xyz"] +
             ["pelvis_height"] + ["touch:" + name for name in ("left", "right", "other")] +
             ["previous_action:" + name for name in info.names])
    assert len(names) == 106
    normalizer = {
        "count": int(buffers["obs_normalizer.count"]), "eps": eps,
        "formula": "(observation - mean) / (stored_std + eps); no post-normalization clipping",
        "minimum_stored_std": float(std.min()), "maximum_stored_std": float(std.max()),
        "minimum_effective_denominator": float((std + eps).min()),
        "maximum_effective_gain": float((1. / (std + eps)).max()),
        "dimensions_std_below_1e_6": int((std < 1e-6).sum()),
        "dimensions_std_below_1e_3": int((std < 1e-3).sum()),
        "narrowest_dimensions": [{"index": int(index), "name": names[index], "mean": float(mean[index]),
                                  "std": float(std[index]), "denominator": float(std[index] + eps)}
                                 for index in np.argsort(std)[:24]],
        "groups": {group: {"min_std": float(std[start:end].min()), "median_std": float(np.median(std[start:end])),
                            "max_std": float(std[start:end].max()), "minimum_denominator": float((std[start:end] + eps).min())}
                   for group,(start,end) in GROUPS.items()},
    }
    arrays = {}
    results = []
    for label, noise_std in (("deterministic", 0.), ("noise_0005", .005)):
        data = mujoco.MjData(info.model)
        reset_deep_crouch(info, data, 7101)
        rng = np.random.default_rng(7101)
        previous = np.zeros(31)
        monitor = TrajectoryLimits(info)
        monitor.observe(data)
        observations, normalized, actions, traces = [], [], [], []
        initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
        reason = "one_second_budget"
        for step in range(50):
            obs = observation_numpy(info, data, previous, sensor_forces(info, data))
            with torch.inference_mode():
                tensor = torch.from_numpy(obs)[None]
                z = actor.obs_normalizer(tensor).numpy().reshape(-1)
                predicted = actor(tensor).numpy().reshape(-1)
            if not np.isfinite(predicted).all() or predicted.shape != (31,):
                raise RuntimeError("Invalid actor output")
            executed = predicted.copy()
            if noise_std:
                executed = executed + rng.normal(0., noise_std, size=31)
            executed = executed.clip(-1., 1.)
            observations.append(obs)
            normalized.append(z)
            actions.append(executed)
            traces.append({"time_s": float(data.time), "pelvis_height_m": float(data.qpos[2]),
                           "max_abs_normalized_observation": float(np.abs(z).max()),
                           "max_normalized_feature": names[int(np.abs(z).argmax())],
                           "max_abs_normalized_previous_action": float(np.abs(z[75:]).max()),
                           "max_abs_previous_action": float(np.abs(previous).max()),
                           "max_abs_actor_output": float(np.abs(predicted).max()),
                           "max_abs_executed_action": float(np.abs(executed).max()),
                           "actor_outputs_outside_action_bounds": int((np.abs(predicted) > 1.).sum()),
                           "normalized_dimensions_above_5": int((np.abs(z) > 5.).sum())})
            target = info.targets(executed)
            for _ in range(info.substeps):
                data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], target)
                mujoco.mj_step(info.model, data)
                monitor.observe(data)
            previous = executed.copy()
            if not monitor.ok:
                reason = "limit_violation"
                break
            if not np.isfinite(data.qpos).all():
                reason = "nonfinite_state"
                break
            if data.qpos[2] < .3:
                reason = "pelvis_below_0.3m"
                break
        obs = observation_numpy(info, data, previous, sensor_forces(info, data))
        with torch.inference_mode():
            terminal_z = actor.obs_normalizer(torch.from_numpy(obs)[None]).numpy().reshape(-1)
        obs_array, z_array = np.asarray(observations), np.asarray(normalized)
        row = {
            "label": label, "seed": 7101, "noise_std": noise_std, "sim_time_s": float(data.time),
            "end_reason": reason, "terminal_height_m": float(data.qpos[2]),
            "trajectory_limits": monitor.report(), "control_samples": len(actions),
            "normalized_pre_action_group_peaks": {group: float(np.abs(z_array[:, start:end]).max())
                                                   for group,(start,end) in GROUPS.items()},
            "terminal_normalized_group_peaks": {group: float(np.abs(terminal_z[start:end]).max())
                                                  for group,(start,end) in GROUPS.items()},
            "peak_normalized_dimensions": [{"index": int(index), "name": names[index],
                                            "max_abs_z": float(np.abs(z_array[:, index]).max())}
                                           for index in np.argsort(np.abs(z_array).max(0))[-15:][::-1]],
            "trace": traces,
        }
        results.append(row)
        arrays.update({label + "_observations": obs_array, label + "_normalized_observations": z_array,
                       label + "_executed_actions": np.asarray(actions), label + "_initial_qpos": initial_qpos,
                       label + "_initial_qvel": initial_qvel})
        print(json.dumps({key: row[key] for key in ("label", "sim_time_s", "end_reason", "terminal_height_m",
                                                   "normalized_pre_action_group_peaks", "trajectory_limits")}), flush=True)
    assert np.array_equal(arrays["deterministic_initial_qpos"], arrays["noise_0005_initial_qpos"])
    assert np.array_equal(arrays["deterministic_initial_qvel"], arrays["noise_0005_initial_qvel"])
    n = min(len(arrays["deterministic_normalized_observations"]), len(arrays["noise_0005_normalized_observations"]))
    comparison = {
        "matched_initial_state": True, "matched_pre_action_steps": n,
        "common_prefix_last_pre_action_time_s": (n - 1) * .02,
        "group_max_abs_normalized_difference": {
            group: float(np.abs(arrays["noise_0005_normalized_observations"][:n, start:end] -
                               arrays["deterministic_normalized_observations"][:n, start:end]).max())
            for group,(start,end) in GROUPS.items()},
    }
    np.savez_compressed(OUT / "paired_observations.npz", **arrays, mean=mean, std=std, feature_names=np.asarray(names))
    summary = {
        "task": "One paired deep-crouch action-noise/normalization diagnostic; no training or correction",
        "actor_sha256": sha256(ACTOR), "checkpoint_sha256": sha256(CHECKPOINT),
        "source_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
                          (Path(__file__), ROOT / "src/x2_recovery/physics.py", ROOT / "src/x2_recovery/common.py")},
        "normalizer": normalizer, "paired_results": results, "comparison": comparison,
        "interpretation": "Noise exposure and normalized feature excursions are correlated with the observed trajectory. This is not a normalization intervention and cannot establish that observation clipping would repair stability.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"normalizer": {key: normalizer[key] for key in ("eps", "minimum_stored_std", "minimum_effective_denominator",
                      "maximum_effective_gain", "dimensions_std_below_1e_6", "dimensions_std_below_1e_3")}, "comparison": comparison}), flush=True)


if __name__ == "__main__":
    main()
