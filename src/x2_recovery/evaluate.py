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
    parser.add_argument("--output", default="artifacts/evaluation")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = RecoveryRuntime(args.controller, args.checkpoint, args.render, args.seed)
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
            try:
                for step in range(round(args.seconds/CONTROL_DT)):
                    start = time.monotonic()
                    state = runtime.step()
                    if step % 5 == 0:
                        history.append({k: v for k,v in state.items() if k not in {"joint_names","joint_positions"}})
                    if writer and step % 2 == 0:
                        renderer.update_scene(runtime.data, camera=camera)
                        writer.append_data(renderer.render())
                    if args.render:
                        time.sleep(max(0., CONTROL_DT-(time.monotonic()-start)))
                    if state["success"] or state["invalid"]:
                        break
            finally:
                if writer: writer.close()
            result = {"episode": episode+1, "seed": args.seed+episode,
                      "outcome": "SUCCEEDED" if state["success"] else "FAILED",
                      "reason": "stable_standing" if state["success"] else "invalid_simulation" if state["invalid"] else "timeout",
                      **{k: v for k,v in state.items() if k not in {"joint_names","joint_positions"}}}
            results.append(result)
            (output/f"episode_{episode+1}.json").write_text(json.dumps({"result":result,"trajectory":history},indent=2))
            print(json.dumps(result),flush=True)
    finally:
        if renderer: renderer.close()
        runtime.close()
    summary = {"controller":args.controller,"checkpoint":args.checkpoint,"episodes":len(results),
               "successes":sum(r["success"] for r in results),"results":results,
               "success_spec":{"hold_seconds":HOLD_SECONDS,"tilt_degrees":float(SUCCESS_TILT*180/3.141592653589793),
                               "pelvis_height_min":runtime.info.height_threshold,"ground_support":"both feet only"}}
    (output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__ == "__main__": main()
