#!/usr/bin/env python3
"""Run the submitted policy, with an optional live viewer or recorded video."""
import argparse
import copy
import json
from pathlib import Path
import time
import threading

import mujoco
from x2_recovery.common import ROOT
from x2_recovery.runtime import RecoveryRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actor', type=Path, default=ROOT/'artifacts/submission/final_recovery/actor.pt')
    parser.add_argument('--seed', type=int, default=30001)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--keep-open', action='store_true',
                        help='Continue physical control after 15 seconds; R restarts the recovery')
    parser.add_argument('--once', dest='keep_open', action='store_false',
                        help='Stop after the original 15-second episode (also use for video)')
    parser.set_defaults(keep_open=False)
    parser.add_argument('--video', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/local_demo/final_policy.json')
    args = parser.parse_args()
    if args.keep_open and not args.render:
        parser.error('--keep-open requires --render')
    if args.keep_open and args.video:
        parser.error('Use --once with --video to record a bounded 15-second episode')
    restart = threading.Event()

    def on_key(keycode):
        # The GUI thread only signals; reset physics on the simulation thread.
        if keycode in (ord('R'), ord('r')):
            restart.set()

    runtime = RecoveryRuntime('full_recovery', args.actor, render=args.render,
                              seed=args.seed, physics_profile='guarded_v2',
                              key_callback=on_key if args.keep_open else None)
    renderer = writer = None
    state = runtime.snapshot(False, False)
    steps = 0
    assessment = None
    paused_failure = False
    end_reason = 'episode_complete'
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
        if args.keep_open:
            print('Live recovery: continues standing after 15 s. Focus the simulator and press R '
                  'to restart; close the window or press Ctrl+C to stop.', flush=True)
        while True:
            if args.render and not runtime.viewer.is_running():
                end_reason = 'window_closed'
                break
            if args.keep_open and restart.is_set():
                restart.clear()
                state = runtime.reset(args.seed)
                steps, assessment, paused_failure = 0, None, False
                end_reason = 'episode_complete'
                start = time.monotonic()
                print('Restarting recovery with seed '+str(args.seed), flush=True)
            if paused_failure:
                # Preserve the failed state for inspection, rather than hiding
                # the fault or continuing invalid dynamics. R remains available.
                runtime.viewer.sync(state_only=True)
                time.sleep(.02)
                continue
            state = runtime.step()
            steps += 1
            if writer and steps % 2 == 0:
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
                if steps == 750:
                    frame.save(args.video.with_suffix('.png'))
            if state['invalid'] or not runtime.limits.ok:
                end_reason = 'physical_failure'
                if not args.keep_open:
                    break
                paused_failure = True
                print('Physical check failed; simulation paused. Press R to restart or close '
                      'the window. This is not a successful standing hold.', flush=True)
                continue
            if steps == 750:
                assessment = copy.deepcopy(state)
                assessment['passed'] = runtime.clean_hold_time >= 2.-1e-8
                print(json.dumps({'assessment_15s_passed': assessment['passed'],
                                  'hold': runtime.clean_hold_time}), flush=True)
                if not args.keep_open:
                    break
                print('Continuing live physics and policy feedback. R restarts; close to exit.',
                      flush=True)
            if args.render:
                time.sleep(max(0., start+steps*.02-time.monotonic()))
    except KeyboardInterrupt:
        end_reason = 'interrupted'
    finally:
        if writer:
            writer.close()
        if renderer:
            renderer.close()
        runtime.close()
    passed = (steps >= 750 and not state['invalid'] and runtime.limits.ok
              and runtime.clean_hold_time >= 2.-1e-8)
    state.update(seed=args.seed, passed=passed, controller='full_recovery',
                 end_reason=end_reason, assessment_15s=assessment)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps({'passed': passed, 'seed': args.seed, 'hold': runtime.clean_hold_time}))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
