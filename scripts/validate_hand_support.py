#!/usr/bin/env python3
"""Validate a scripted hands-supported squat return, never floor recovery."""
import argparse
import hashlib
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from screen_supported_lowering import run, target
from x2_recovery.common import CONTROL_DT, MODEL_PATH, ModelInfo, ground_forces, success_conditions
from x2_recovery.physics import reset_deep_crouch, deep_crouch_target, TrajectoryLimits
from x2_recovery.stance import stance_metrics


def exact_contacts(info, data):
    values = {}
    force = np.empty(6)
    for index in range(data.ncon):
        c = data.contact[index]
        if c.geom1 in info.floor_geoms:
            other = c.geom2
        elif c.geom2 in info.floor_geoms:
            other = c.geom1
        else:
            continue
        mujoco.mj_contactForce(info.model, data, index, force)
        if force[0] > 0.:
            name = info.model.body(int(info.model.geom_bodyid[other])).name
            values[name] = values.get(name, 0.) + float(force[0])
    return values


def caption(frame, time_s, height):
    # Captions sit outside the rendered viewport so they never hide the robot.
    img = Image.new("RGB", (640, 592), (16, 23, 31))
    img.paste(Image.fromarray(frame), (0, 80))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default(size=16)
    draw.text((12, 7), "SCRIPTED INTERMEDIATE | NOT LEARNED", fill=(255, 220, 95), font=font)
    draw.text((12, 29), "Deep crouch -> hands-supported squat -> standing", fill="white", font=font)
    draw.text((12, 51), f"Simulation {time_s:5.2f} s | pelvis {height:.3f} m", fill="white", font=font)
    draw.text((12, 568), "NOT FLOOR RECOVERY - physical MuJoCo re-simulation", fill=(255, 220, 95), font=font)
    return np.asarray(img)


def recorded_run(info, seed, expected_min_time, output):
    data = mujoco.MjData(info.model)
    reset_deep_crouch(info, data, seed)
    monitor = TrajectoryLimits(info)
    monitor.observe(data)
    low, start = target(info, 2.32, .9), deep_crouch_target(info)
    times = [0., 4., 6., 10., 13., 16.]
    poses = [start, low, low, start, info.nominal, info.nominal]
    renderer = mujoco.Renderer(info.model, height=480, width=640)
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [-.1, 0., .6]
    camera.distance, camera.azimuth, camera.elevation = 2.7, 125, -18
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0
    minimum = float(data.qpos[2])
    hold = max_hold = 0.
    max_clamp = 0.
    body_set = set()
    rows = []
    bottom_saved = False
    reason = "duration"
    frame_count = 0

    def image():
        renderer.update_scene(data, camera=camera, scene_option=options)
        return caption(renderer.render(), float(data.time), float(data.qpos[2]))

    try:
        imageio.imwrite(output / "start.png", image())
        with imageio.get_writer(output / "scripted_hand_support.mp4", fps=25) as writer:
            for step in range(round(16. / CONTROL_DT)):
                time_s = step * CONTROL_DT
                index = min(max(np.searchsorted(times, time_s, side="right") - 1, 0), len(times) - 2)
                phase = np.clip((time_s - times[index]) / (times[index + 1] - times[index]), 0., 1.)
                phase = phase * phase * (3. - 2. * phase)
                command = poses[index] * (1. - phase) + poses[index + 1] * phase
                max_clamp = max(max_clamp, float(np.abs(command - np.clip(command, info.lower + .05, info.upper - .05)).max()))
                for _ in range(info.substeps):
                    data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], command)
                    mujoco.mj_step(info.model, data)
                    monitor.observe(data)
                    if not monitor.ok:
                        reason = "limit_violation"
                        break
                support = exact_contacts(info, data)
                body_set.update(support)
                checks = success_conditions(info, data, ground_forces(info, data))
                stance = stance_metrics(info, data)
                clean = all(checks.values()) and stance["posture_ok"] and monitor.ok
                hold = hold + CONTROL_DT if clean else 0.
                max_hold = max(max_hold, hold)
                minimum = min(minimum, float(data.qpos[2]))
                rows.append({"time_s": float(data.time), "pelvis_height_m": float(data.qpos[2]),
                             "torso_up_cosine": float(data.xmat[info.torso_id].reshape(3, 3)[2, 2]),
                             "exact_ground_normal_forces_by_body_n": support,
                             "base_linear_speed_m_s": float(np.linalg.norm(data.qvel[:3])),
                             "base_angular_speed_rad_s": float(np.linalg.norm(data.qvel[3:6])),
                             "clean_standing": bool(clean), "standing_checks": checks})
                if not bottom_saved and abs(float(data.time) - expected_min_time) < 1e-7:
                    imageio.imwrite(output / "bottom.png", image())
                    bottom_saved = True
                if step % 2 == 1:
                    writer.append_data(image())
                    frame_count += 1
                if reason != "duration":
                    break
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or data.qpos[2] < .05:
                    reason = "invalid_state"
                    break
            imageio.imwrite(output / "end.png", image())
    finally:
        renderer.close()
    complete = abs(float(data.time) - 16.) < 1e-6
    return {
        "scope": "SCRIPTED INTERMEDIATE: deep crouch, hands-supported squat, standing return; not learned or floor recovery",
        "seed": seed, "complete": complete, "reason": reason, "sim_time_s": float(data.time),
        "minimum_height_m": minimum, "terminal_clean_hold_s": hold, "max_clean_hold_s": max_hold,
        "return_clean_pass": bool(complete and hold >= 2. - 1e-8 and monitor.ok),
        "target_clipping_rad": max_clamp, "limits": monitor.report(),
        "ground_contact_bodies": sorted(body_set), "final": rows[-1],
        "video_frames": frame_count, "video_fps": 25, "bottom_frame_saved": bottom_saved,
        "final_qpos": data.qpos.tolist(), "final_qvel": data.qvel.tolist(), "trajectory_50hz": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/validation/floor_transition/hand_support_validation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        parser.error("Choose a fresh output directory")
    info = ModelInfo(physics_profile="guarded_v2")
    root = Path(__file__).resolve().parents[1]
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    config = {
        "scope": "Independent scripted hands-supported squat and standing-return validation; not learned or floor recovery",
        "source_function": "screen_supported_lowering.run(info,2.32,0.9,seed)",
        "validation_seeds": list(range(9101, 9106)), "recorded_resimulation_seed": 9101,
        "physics_profile": "guarded_v2", "physics_dt_s": float(info.model.opt.timestep),
        "control_dt_s": CONTROL_DT, "episode_seconds": 16., "stage_times_s": [0., 4., 6., 10., 13., 16.],
        "start": "reset_deep_crouch, physically lowered from standing; no later state injection",
        "monitoring": {"joint_position_speed_commanded_effort_hz": 1000, "exact_contacts_and_posture_hz": 50},
        "pass_definition": "Full16s and>=2s clean terminal standing with no latched trajectory-limit violation",
        "source_sha256": {str(path.relative_to(root)): sha(path) for path in
                          (Path(__file__), root / "scripts/screen_supported_lowering.py", MODEL_PATH,
                           root / "src/x2_recovery/common.py", root / "src/x2_recovery/physics.py", root / "src/x2_recovery/stance.py")},
    }
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results = []
    for seed in range(9101, 9106):
        result = run(info, 2.32, .9, seed)
        results.append(result)
        (args.output / f"validation_{seed}.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: result[key] for key in ("seed", "return_clean_pass", "minimum_height",
                                                      "terminal_clean_hold", "limits")}), flush=True)
    recorded = recorded_run(info, 9101, results[0]["minimum_state"]["time"], args.output)
    crosscheck = {
        "same_pass_outcome": recorded["return_clean_pass"] == results[0]["return_clean_pass"],
        "minimum_height_matches": abs(recorded["minimum_height_m"] - results[0]["minimum_height"]) < 1e-10,
        "final_height_matches": abs(recorded["final"]["pelvis_height_m"] - results[0]["final"]["pelvis_height"]) < 1e-10,
        "terminal_hold_matches": abs(recorded["terminal_clean_hold_s"] - results[0]["terminal_clean_hold"]) < 1e-10,
        "limit_report_matches": recorded["limits"] == results[0]["limits"],
        "same_ground_contact_bodies": recorded["ground_contact_bodies"] == results[0]["ground_contact_bodies"],
        "video_is_16_seconds": recorded["video_frames"] == 400 and recorded["video_fps"] == 25,
        "bottom_frame_saved": recorded["bottom_frame_saved"], "zero_target_clipping": recorded["target_clipping_rad"] <= 1e-10,
    }
    (args.output / "recorded_episode.json").write_text(json.dumps(recorded, indent=2) + "\n")
    summary = {
        "scope": config["scope"], "validation_episodes": 5,
        "scripted_return_passes": sum(result["return_clean_pass"] for result in results),
        "trajectory_limit_passes": sum(result["limits"]["ok"] for result in results),
        "minimum_height_range_m": [min(r["minimum_height"] for r in results), max(r["minimum_height"] for r in results)],
        "terminal_clean_hold_range_s": [min(r["terminal_clean_hold"] for r in results), max(r["terminal_clean_hold"] for r in results)],
        "recorded_resimulation_crosscheck": crosscheck,
        "video_sha256": sha(args.output / "scripted_hand_support.mp4"),
        "results": results,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key,value in summary.items() if key != "results"}), flush=True)
    if not all(crosscheck.values()):
        raise RuntimeError(f"Recorded resimulation differs: {crosscheck}")


if __name__ == "__main__":
    main()
