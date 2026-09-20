#!/usr/bin/env python3
"""Run the submitted policy, with an optional live viewer or recorded video."""
import argparse
import copy
import json
from pathlib import Path
import time

import mujoco
from x2_recovery.common import ROOT
from x2_recovery.runtime import RecoveryRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', type=Path, default=ROOT/'artifacts/submission/final_recovery/actor.pt')
    parser.add_argument('--seed', type=int, default=30001)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/local_demo/final_policy.json')
    args = parser.parse_args()
    runtime = RecoveryRuntime('full_recovery', args.actor, render=args.render,
                              seed=args.seed, physics_profile='guarded_v2')
    renderer = writer = None
    try:
        if args.video:
            import imageio.v2 as imageio
            from PIL import Image, ImageDraw, ImageFont
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=25, codec='libx264', quality=8)
            view_model = copy.copy(runtime.model)
            view_data = mujoco.MjData(view_model)
            renderer = mujoco.Renderer(view_model, height=720, width=960)
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 3.2, 135, -20
            camera.lookat[:] = [0., 0., .55]
            font = ImageFont.load_default(size=19)
            option = mujoco.MjvOption()
            option.geomgroup[3] = 0
        start = time.monotonic()
        for step in range(750):
            state = runtime.step()
            if writer and step % 2 == 1:
                # Rendering owns separate data and cannot change policy sensors or physics.
                mujoco.mj_copyData(view_data, view_model, runtime.data)
                renderer.update_scene(view_data, camera=camera, scene_option=option)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 960, 62), fill=(18, 23, 28))
                draw.text((18, 10), 'AgiBot X2 | motion prior + PPO feedback | seed '+str(args.seed), fill='white', font=font)
                draw.text((18, 34), f"t={state['sim_time']:.2f}s   clean standing={runtime.clean_hold_time:.2f}/2.00s   limits={'PASS' if runtime.limits.ok else 'FAIL'}", fill='white', font=font)
                import numpy as np
                writer.append_data(np.asarray(frame))
                if step == 749:
                    frame.save(args.video.with_suffix('.png'))
            if state['invalid'] or not runtime.limits.ok:
                break
            if args.render:
                time.sleep(max(0., start+(step+1)*.02-time.monotonic()))
        passed = step == 749 and runtime.limits.ok and runtime.clean_hold_time >= 2.-1e-8
        state.update(seed=args.seed, passed=passed, controller='full_recovery')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(state, indent=2)+'\n')
        print(json.dumps({'passed': passed, 'seed': args.seed, 'hold': runtime.clean_hold_time}))
        return 0 if passed else 1
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
        runtime.close()


if __name__ == '__main__':
    raise SystemExit(main())
