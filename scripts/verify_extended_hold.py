#!/usr/bin/env python3
"""Hold the recovered stance headlessly for a fixed simulated duration.

RUN_DEMO.sh --keep-open needs a display and runs until the window closes, so it
cannot record how long the policy stays up. This drives the same runtime without
a viewer and stops at the first physical fault, if any.
"""
import argparse
import json
from pathlib import Path
from x2_recovery.common import ROOT
from x2_recovery.runtime import RecoveryRuntime

CONTROL_DT = .02

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor', type=Path, default=ROOT/'artifacts/submission/final_recovery/actor.pt')
    parser.add_argument('--seed', type=int, default=30001)
    parser.add_argument('--seconds', type=float, default=120.)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    steps_target = round(args.seconds/CONTROL_DT)
    runtime = RecoveryRuntime('full_recovery', args.actor, render=False,
                              seed=args.seed, physics_profile='guarded_v2')
    state = runtime.snapshot(False, False)
    fault = None
    hold_at_assessment = None
    for step in range(1, steps_target+1):
        state = runtime.step()
        if state['invalid'] or not runtime.limits.ok:
            fault = {'step': step, 'sim_time': state['sim_time'],
                     'invalid': bool(state['invalid']), 'limits_ok': bool(runtime.limits.ok)}
            break
        if step == 750:
            hold_at_assessment = runtime.clean_hold_time

    report = {'seed': args.seed, 'requested_sim_seconds': args.seconds,
              'reached_sim_time': state['sim_time'], 'first_fault': fault,
              'clean_hold_at_15s': hold_at_assessment,
              'final_clean_hold_seconds': runtime.clean_hold_time,
              'final_pelvis_height': state.get('pelvis_height'),
              'physics_profile': 'guarded_v2',
              'limits': {'ok': bool(runtime.limits.ok), 'physics_steps': runtime.limits.steps,
                         'max_position_error': runtime.limits.max_position_error,
                         'max_speed_ratio': runtime.limits.max_speed_ratio,
                         'max_effort_ratio': runtime.limits.max_effort_ratio,
                         'first_violation': runtime.limits.first_violation},
              'passed': fault is None and runtime.limits.ok}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'passed': report['passed'], 'reached_sim_time': report['reached_sim_time'],
                      'final_clean_hold_seconds': report['final_clean_hold_seconds']}))
