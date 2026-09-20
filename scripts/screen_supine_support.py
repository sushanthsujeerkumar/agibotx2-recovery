#!/usr/bin/env python3
"""Bounded scripted supine-to-supported-motion screen; not standing/recovery."""
import argparse
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np

from x2_recovery.common import CONTROL_DT, MODEL_PATH, ModelInfo, reset_cpu
from x2_recovery.physics import TrajectoryLimits


def symmetric(hip, knee, ankle, shoulder, elbow, width=.2, waist=.22, leg_width=0.):
    result = {"waist_pitch_joint": waist}
    for side, sign in (("left", 1.), ("right", -1.)):
        for name, value in (("hip_pitch", hip), ("knee", knee), ("ankle_pitch", ankle),
                            ("shoulder_pitch", shoulder), ("elbow", elbow),
                            ("shoulder_roll", sign * width), ("hip_roll", sign * leg_width)):
            result[f"{side}_{name}_joint"] = value
    return result


def mirror(values):
    out = {}
    for name, value in values.items():
        swapped = name.replace("left_", "TEMP_").replace("right_", "left_").replace("TEMP_", "right_")
        out[swapped] = -value if any(axis in name for axis in ("roll", "yaw")) else value
    return out


def candidates():
    rows = []
    for label, hip, knee, ankle, shoulder in (
            ("bilateral_press_moderate", -1.2, 1.8, -.6, .8),
            ("bilateral_press_tucked", -1.6, 2., -.4, 1.),
            ("bilateral_press_deep", -1.8, 2.2, -.4, 1.4)):
        rows.append((label, symmetric(hip, knee, ankle, shoulder, -1.),
                     symmetric(hip, knee, ankle, shoulder, -.12)))
    rows.append(("tuck_then_open_with_arm_press", symmetric(-2., 2.2, -.4, .8, -1.3),
                 symmetric(-.9, 1.6, -.7, 1.5, -.12)))
    roll = {"waist_yaw_joint": -2., "waist_pitch_joint": .2,
            "left_hip_pitch_joint": -1.2, "right_hip_pitch_joint": -.6,
            "left_knee_joint": 1.8, "right_knee_joint": 1.3,
            "left_hip_yaw_joint": -.8, "left_hip_roll_joint": -.15, "right_hip_roll_joint": -.25,
            "left_shoulder_pitch_joint": -.6, "right_shoulder_pitch_joint": 1.8,
            "left_shoulder_yaw_joint": -1.5, "left_shoulder_roll_joint": 0., "right_shoulder_roll_joint": -.2,
            "left_elbow_joint": -1.8, "right_elbow_joint": -.3}
    settle = symmetric(-1.3, 1.9, -.6, .8, -.2)
    settle.update(waist_yaw_joint=-.7, waist_roll_joint=-.2,
                  left_shoulder_pitch_joint=-.3, left_elbow_joint=-1.4,
                  right_shoulder_pitch_joint=1.3, right_elbow_joint=-.12)
    rows.extend((("baseline_side_roll_left", roll, settle),
                 ("baseline_side_roll_right", mirror(roll), mirror(settle))))
    roll2 = dict(roll)
    roll2.update(waist_yaw_joint=-1., waist_roll_joint=-.35,
                 left_shoulder_pitch_joint=-.3, right_shoulder_pitch_joint=1.4,
                 left_shoulder_yaw_joint=-1.2, left_hip_pitch_joint=-1.4,
                 left_knee_joint=2.1, right_hip_pitch_joint=-.8, right_knee_joint=1.4)
    settle2 = dict(settle)
    settle2.update(waist_yaw_joint=-1., waist_roll_joint=-.2,
                   right_shoulder_pitch_joint=1., left_shoulder_pitch_joint=.4)
    rows.extend((("gentle_side_roll_left", roll2, settle2),
                 ("gentle_side_roll_right", mirror(roll2), mirror(settle2))))
    roll3 = dict(roll2)
    roll3.update(waist_yaw_joint=-.6, waist_roll_joint=-.35,
                 left_hip_roll_joint=-.15, right_hip_roll_joint=-.45,
                 left_hip_yaw_joint=-.4, right_hip_yaw_joint=-.4)
    settle3 = symmetric(-1.5, 2.1, -.6, .8, -.2)
    settle3.update(waist_yaw_joint=-.3, waist_roll_joint=-.1,
                   left_shoulder_pitch_joint=.3, left_elbow_joint=-1.4,
                   right_shoulder_pitch_joint=1.4, right_elbow_joint=-.12)
    rows.extend((("lateral_leg_tuck_left", roll3, settle3),
                 ("lateral_leg_tuck_right", mirror(roll3), mirror(settle3))))
    rows.append(("wide_arm_press", symmetric(-1.4, 2.1, -.7, 1.1, -1.2, width=.6, waist=.24),
                 symmetric(-1.4, 2.1, -.7, 1.6, -.12, width=.6, waist=.24)))
    rows.append(("wide_arm_and_leg_press", symmetric(-1.4, 2.1, -.7, 1.2, -1.2, width=1., waist=.24, leg_width=.35),
                 symmetric(-1.4, 2.1, -.7, 1.8, -.15, width=1., waist=.24, leg_width=.35)))
    assert len(rows) == 12
    return rows


def target(info, values):
    q = info.nominal.copy()
    for name, value in values.items():
        q[info.names.index(name)] = value
    if np.any(q < info.lower + .05) or np.any(q > info.upper - .05):
        raise ValueError("Authored target exceeds guarded command margins")
    return q


def exact_contacts(info, data):
    forces = {}
    wrench = np.empty(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 in info.floor_geoms:
            body = int(info.model.geom_bodyid[g2])
        elif g2 in info.floor_geoms:
            body = int(info.model.geom_bodyid[g1])
        else:
            continue
        mujoco.mj_contactForce(info.model, data, index, wrench)
        force = max(float(wrench[0]), 0.)
        if force > 0.:
            name = info.model.body(body).name
            forces[name] = forces.get(name, 0.) + force
    return forces


def snapshot(info, data):
    pelvis = info.model.body("pelvis").id
    return {"time_s": float(data.time), "pelvis_height_m": float(data.qpos[2]),
            "torso_height_m": float(data.xpos[info.torso_id, 2]),
            "pelvis_up_cosine": float(data.xmat[pelvis].reshape(3, 3)[2, 2]),
            "torso_up_cosine": float(data.xmat[info.torso_id].reshape(3, 3)[2, 2]),
            "pelvis_orientation_wxyz": data.xquat[pelvis].tolist(),
            "torso_orientation_wxyz": data.xquat[info.torso_id].tolist(),
            "base_linear_speed_m_s": float(np.linalg.norm(data.qvel[:3])),
            "base_angular_speed_rad_s": float(np.linalg.norm(data.qvel[3:6])),
            "ground_normal_forces_by_body_n": exact_contacts(info, data)}


def rank(state):
    return ((state["torso_up_cosine"] + state["pelvis_up_cosine"]) / 2.,
            min(state["torso_up_cosine"], state["pelvis_up_cosine"]), state["pelvis_height_m"])


def episode(info, candidate, seed):
    name, first, second = candidate
    d = mujoco.MjData(info.model)
    reset_cpu(info, d, seed)
    start = d.qpos[info.qadr].copy()
    if np.any(start < info.lower + .05) or np.any(start > info.upper - .05):
        raise ValueError("Settled initial command lies outside guarded margins")
    poses = [start, target(info, first), target(info, first), target(info, second), target(info, second)]
    times = [0., 3., 5., 8., 12.]
    monitor = TrajectoryLimits(info)
    monitor.observe(d)
    initial = snapshot(info, d)
    peak = initial
    best_static = None
    static_hold = max_static_hold = 0.
    intermediate_hold = max_intermediate_hold = 0.
    history = [initial]
    positions, velocities, controls, frame_times = [d.qpos.copy()], [d.qvel.copy()], [d.ctrl.copy()], [0.]
    max_clamp = 0.
    reason = "duration"
    for step in range(600):
        time_s = step * CONTROL_DT
        segment = min(max(int(np.searchsorted(times, time_s, side="right")) - 1, 0), len(times) - 2)
        phase = (time_s - times[segment]) / (times[segment + 1] - times[segment])
        phase = phase * phase * (3. - 2. * phase)
        command = (1. - phase) * poses[segment] + phase * poses[segment + 1]
        max_clamp = max(max_clamp, float(np.abs(command - np.clip(command, info.lower + .05, info.upper - .05)).max()))
        for _ in range(info.substeps):
            d.ctrl[:] = info.torque(d.qpos[info.qadr], d.qvel[info.vadr], command)
            mujoco.mj_step(info.model, d)
            monitor.observe(d)
            if not monitor.ok:
                reason = "trajectory_limit_fault"
                break
            if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
                reason = "nonfinite_state"
                break
        state = snapshot(info, d)
        history.append(state)
        positions.append(d.qpos.copy())
        velocities.append(d.qvel.copy())
        controls.append(d.ctrl.copy())
        frame_times.append(float(d.time))
        if rank(state) > rank(peak):
            peak = state
        supported_static = (sum(state["ground_normal_forces_by_body_n"].values()) > 20. and
                            state["base_linear_speed_m_s"] < .10 and state["base_angular_speed_rad_s"] < .30 and monitor.ok)
        static_hold = static_hold + CONTROL_DT if supported_static else 0.
        max_static_hold = max(max_static_hold, static_hold)
        if static_hold >= 1. - 1e-8 and (best_static is None or rank(state) > rank(best_static)):
            best_static = state
        intermediate = (supported_static and state["torso_up_cosine"] >= .5 and
                        state["pelvis_up_cosine"] >= .3)
        intermediate_hold = intermediate_hold + CONTROL_DT if intermediate else 0.
        max_intermediate_hold = max(max_intermediate_hold, intermediate_hold)
        if reason != "duration":
            break
    complete = abs(float(d.time) - 12.) < 1e-6
    intermediate_reached = complete and monitor.ok and max_clamp <= 1e-10 and max_intermediate_hold >= 1. - 1e-8
    promising = (complete and monitor.ok and best_static is not None and
                 best_static["torso_up_cosine"] >= .5 and best_static["pelvis_up_cosine"] >= .3)
    row = {"name": name, "seed": seed, "scope": "supported intermediate only; not standing or recovery",
           "complete": complete, "end_reason": reason, "sim_time_s": float(d.time),
           "target_clipping_rad": max_clamp, "trajectory_limits": monitor.report(),
           "initial": initial, "peak_orientation_state": peak, "best_supported_static_state": best_static,
           "final": state, "max_supported_static_hold_s": max_static_hold,
           "max_supported_intermediate_hold_s": max_intermediate_hold,
           "supported_intermediate_reached": intermediate_reached, "promising_supported_motion": bool(promising)}
    arrays = dict(qpos=np.asarray(positions), qvel=np.asarray(velocities), ctrl=np.asarray(controls), times_s=np.asarray(frame_times))
    return row, history, arrays


def reassess_supported_motion(row, history):
    """Reclassify recorded states after dropping a height-based sitting filter."""
    row.setdefault("initial_higher_pelvis_classification", {
        "supported_intermediate_reached": row["supported_intermediate_reached"],
        "promising_supported_motion": row["promising_supported_motion"],
        "max_supported_intermediate_hold_s": row["max_supported_intermediate_hold_s"],
    })
    hold = maximum = 0.
    for state in history[1:]:
        eligible = (sum(state["ground_normal_forces_by_body_n"].values()) > 20. and
                    state["base_linear_speed_m_s"] < .10 and state["base_angular_speed_rad_s"] < .30 and
                    state["torso_up_cosine"] >= .5 and state["pelvis_up_cosine"] >= .3 and
                    row["trajectory_limits"]["ok"])
        hold = hold + CONTROL_DT if eligible else 0.
        maximum = max(maximum, hold)
    best = row["best_supported_static_state"]
    row["max_supported_intermediate_hold_s"] = maximum
    row["supported_intermediate_reached"] = bool(row["complete"] and row["trajectory_limits"]["ok"] and
                                                row["target_clipping_rad"] <= 1e-10 and maximum >= 1. - 1e-8)
    row["promising_supported_motion"] = bool(row["complete"] and row["trajectory_limits"]["ok"] and best is not None and
                                             best["torso_up_cosine"] >= .5 and best["pelvis_up_cosine"] >= .3)
    row["classification_note"] = "Pelvis height retained as a measurement; it does not disqualify a supported sitting/propped intermediate."


def render_recorded_states(info, arrays, directory):
    """Visualize recorded physics states only; this replay does not run dynamics."""
    import imageio.v2 as imageio
    renderer = mujoco.Renderer(info.model, height=480, width=640)
    visual = mujoco.MjData(info.model)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.15, 0., .3]
    camera.distance, camera.azimuth, camera.elevation = 2.3, 125, -22
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    frames = sorted(set([0, len(arrays["qpos"]) // 3, 2 * len(arrays["qpos"]) // 3, len(arrays["qpos"]) - 1]))
    try:
        with imageio.get_writer(directory / "recorded_motion.mp4", fps=25) as writer:
            for index in range(len(arrays["qpos"])):
                if index % 2 and index not in frames:
                    continue
                visual.qpos[:] = arrays["qpos"][index]
                visual.qvel[:] = arrays["qvel"][index]
                visual.ctrl[:] = arrays["ctrl"][index]
                mujoco.mj_forward(info.model, visual)
                renderer.update_scene(visual, camera=camera, scene_option=options)
                image = renderer.render()
                if index % 2 == 0:
                    writer.append_data(image)
                if index in frames:
                    imageio.imwrite(directory / f"frame_{index:04d}.png", image)
    finally:
        renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation/floor_transition/supine_support"))
    parser.add_argument("--reassess-existing", action="store_true",
                        help="Reclassify the saved 12-candidate screen without rerunning its physics, then validate the selected intermediate.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists() and not args.reassess_existing:
        parser.error("Choose a fresh output directory")
    if args.reassess_existing and any(args.output.glob("validation_*.json")):
        parser.error("Fresh-seed validation already exists; do not repeat the bounded experiment")
    info = ModelInfo(physics_profile="guarded_v2")
    choices = candidates()
    for _, first, second in choices:
        target(info, first)
        target(info, second)
    root = Path(__file__).resolve().parents[1]
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    config = {"scope": "scripted supine-to-supported intermediate feasibility; not standing or recovery",
              "physics_profile": "guarded_v2", "development_seed": 9001, "validation_seeds_if_promising": list(range(9101, 9106)),
              "max_candidates": 12, "episode_duration_s": 12., "stage_times_s": [0., 3., 5., 8., 12.],
              "candidates": [dict(name=name, first=first, second=second) for name,first,second in choices],
              "monitoring": {"joint_position_speed_commanded_effort_hz": 1000, "exact_ground_contact_and_orientation_hz": 50},
              "static_spec": {"ground_force_total_min_n": 20., "base_linear_speed_max_m_s": .10, "base_angular_speed_max_rad_s": .30, "hold_seconds": 1.},
              "intermediate_spec": {"torso_up_cosine_min": .5, "pelvis_up_cosine_min": .3, "pelvis_height_threshold": None,
                                    "standing_required": False, "nonfoot_ground_support_allowed": True},
              "selection": "Complete limit-valid supported-static candidates with torso-up>=0.5 and pelvis-up>=0.3; rank average uprightness, then worst uprightness, then measured pelvis height. Low pelvis does not disqualify sitting. Reward is not used.",
              "source_sha256": {str(path.relative_to(root)): sha(path) for path in
                                (Path(__file__), MODEL_PATH, root / "src/x2_recovery/common.py", root / "src/x2_recovery/physics.py")}}
    if args.reassess_existing:
        previous_config = json.loads((args.output / "config.json").read_text())
        config["original_screen_source_sha256"] = previous_config["source_sha256"]
        config["classification_revision"] = "After the user-directed review, remove pelvis-height eligibility and reassess saved exact-state/contact histories. No original screen simulation repeated."
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    screen, records = [], {}
    for index, candidate in enumerate(choices):
        if args.reassess_existing:
            saved = json.loads((args.output / f"screen_{index:02d}.json").read_text())
            row, history = saved["result"], saved["trajectory"]
            arrays = dict(np.load(args.output / f"screen_{index:02d}_states.npz"))
            reassess_supported_motion(row, history)
        else:
            row, history, arrays = episode(info, candidate, 9001)
        row["candidate_index"] = index
        screen.append(row)
        records[index] = arrays
        (args.output / f"screen_{index:02d}.json").write_text(json.dumps(dict(result=row, trajectory=history), indent=2) + "\n")
        np.savez_compressed(args.output / f"screen_{index:02d}_states.npz", **arrays)
        print(json.dumps({"candidate": index, "name": row["name"], "complete": row["complete"], "reason": row["end_reason"],
                          "intermediate_reached": row["supported_intermediate_reached"], "promising": row["promising_supported_motion"],
                          "peak_torso_up": row["peak_orientation_state"]["torso_up_cosine"],
                          "best_static": row["best_supported_static_state"], "limits": row["trajectory_limits"]}), flush=True)
    promising = [row for row in screen if row["promising_supported_motion"]]
    best = max(promising, key=lambda row: rank(row["best_supported_static_state"])) if promising else None
    validation = []
    if best is not None:
        video = args.output / "selected_recording"
        video.mkdir(exist_ok=True)
        render_recorded_states(info, records[best["candidate_index"]], video)
        for seed in range(9101, 9106):
            row, history, arrays = episode(info, choices[best["candidate_index"]], seed)
            validation.append(row)
            (args.output / f"validation_{seed}.json").write_text(json.dumps(dict(result=row, trajectory=history), indent=2) + "\n")
            print(json.dumps({"validation_seed": seed, "name": row["name"], "intermediate_reached": row["supported_intermediate_reached"],
                              "promising": row["promising_supported_motion"], "limits": row["trajectory_limits"]}), flush=True)
    summary = {"scope": config["scope"], "screened_candidates": len(screen),
               "complete_limit_valid_candidates": sum(row["complete"] and row["trajectory_limits"]["ok"] for row in screen),
               "supported_intermediate_candidates": sum(row["supported_intermediate_reached"] for row in screen),
               "selected_candidate": best, "validation_episodes": len(validation),
               "validation_supported_intermediates": sum(row["supported_intermediate_reached"] for row in validation),
               "screen_results": screen, "validation_results": validation}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key,value in summary.items() if key not in ("screen_results", "validation_results", "selected_candidate")}), flush=True)


if __name__ == "__main__":
    main()
