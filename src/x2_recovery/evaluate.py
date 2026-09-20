"""Five fixed-seed evaluation episodes, optional viewer and MP4 recording."""
import argparse
import json
import time
from pathlib import Path
from .runtime import RecoveryRuntime
from .common import CONTROL_DT, EPISODE_SECONDS, HOLD_SECONDS, SUCCESS_TILT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", choices=["scripted", "policy"], default="scripted")
    parser.add_argument("--checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--seconds", type=float, default=EPISODE_SECONDS)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--assess-stance", action="store_true", help="Report extra posture criteria and continue after original recovery until clean stance or timeout.")
    parser.add_argument("--physics-profile", choices=["legacy", "guarded_v2"], default="legacy")
    parser.add_argument("--require-limits", action="store_true", help="Fail immediately after any trajectory limit violation.")
    parser.add_argument("--output", default="artifacts/evaluation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = RecoveryRuntime(args.controller, args.checkpoint, args.render, args.seed,
                              assess_stance=args.assess_stance, physics_profile=args.physics_profile)
    renderer = None
    if args.video:
        import mujoco
        import imageio.v2 as imageio
        renderer = mujoco.Renderer(runtime.model, height=480, width=640)
        camera = mujoco.MjvCamera()
        camera.lookat[:] = [0, 0, .45]
        camera.distance, camera.azimuth, camera.elevation = 2.5, 135, -20
    results = []
    try:
        for episode in range(args.episodes):
            runtime.reset(args.seed+episode)
            writer = imageio.get_writer(output/f"episode_{episode+1}.mp4", fps=25) if renderer else None
            history = []
            first_recovery_time = None
            try:
                for step in range(round(args.seconds/CONTROL_DT)):
                    start = time.monotonic()
                    state = runtime.step()
                    if state['success'] and first_recovery_time is None:
                        first_recovery_time = state['sim_time']
                    if step % 5 == 0:
                        history.append({k: v for k,v in state.items() if k not in {"joint_names","joint_positions"}})
                    if writer and step % 2 == 0:
                        renderer.update_scene(runtime.data, camera=camera)
                        writer.append_data(renderer.render())
                    if args.render:
                        time.sleep(max(0., CONTROL_DT-(time.monotonic()-start)))
                    passed = state['clean_stance_success'] if args.assess_stance else state['success']
                    if passed or state["invalid"] or (args.require_limits and not state['trajectory_limits']['ok']):
                        break
            finally:
                if writer: writer.close()
            result = {"episode": episode+1, "seed": args.seed+episode,
                      "outcome": "SUCCEEDED" if state["success"] else "FAILED",
                      "reason": "stable_standing" if state["success"] else "invalid_simulation" if state["invalid"] else "timeout",
                      **{k: v for k,v in state.items() if k not in {"joint_names","joint_positions"}}}
            results.append(result)
            if args.require_limits and not state['trajectory_limits']['ok']:
                result.update(success=False, outcome='FAILED', reason='trajectory_limit_violation')
            if args.assess_stance:
                accepted = first_recovery_time is not None and (not args.require_limits or state['trajectory_limits']['ok'])
                result.update(success=accepted, recovery_time_s=first_recovery_time,
                              outcome='SUCCEEDED' if accepted else 'FAILED',
                              reason='stable_standing' if accepted else result['reason'],
                              clean_stance_outcome='SUCCEEDED' if state['clean_stance_success'] else 'FAILED')
            (output/f"episode_{episode+1}.json").write_text(json.dumps({"result":result,"trajectory":history},indent=2))
            print(json.dumps(result),flush=True)
    finally:
        if renderer: renderer.close()
        runtime.close()
    summary = {"controller":args.controller,"checkpoint":args.checkpoint,"episodes":len(results),
               "successes":sum(r["success"] for r in results),"results":results,
               "success_spec":{"hold_seconds":HOLD_SECONDS,"tilt_degrees":float(SUCCESS_TILT*180/3.141592653589793),
                               "pelvis_height_min":runtime.info.height_threshold,"ground_support":"both feet only"}}
    if args.assess_stance:
        from .stance import MIN_WIDTH, MAX_WIDTH, MAX_HEADING, MAX_HIP_YAW, MAX_SOLE_TILT, MAX_FOOT_BRACING_FORCE
        summary['clean_stance_successes'] = sum(r['clean_stance_success'] for r in results)
        summary['clean_stance_spec'] = {'hold_seconds': HOLD_SECONDS, 'original_standing_required': True,
                                      'signed_width_m': [MIN_WIDTH, MAX_WIDTH], 'max_heading_rad': MAX_HEADING,
                                      'max_hip_yaw_rad': MAX_HIP_YAW, 'max_sole_tilt_rad': MAX_SOLE_TILT,
                                      'max_foot_bracing_force_n': MAX_FOOT_BRACING_FORCE}
    summary.update(physics_profile=args.physics_profile, require_limits=args.require_limits,
                   trajectory_limit_passes=sum(r['trajectory_limits']['ok'] for r in results),
                   validated_successes=sum(r['validated_success'] for r in results))
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__ == "__main__": main()
