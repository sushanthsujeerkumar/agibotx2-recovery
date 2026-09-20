#!/usr/bin/env python3
"""Read-only audit at every physics step; does not clamp or modify robot state."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from x2_recovery.runtime import RecoveryRuntime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='artifacts/submission/local_stability/actor.pt')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--seed', type=int, default=1001)
    parser.add_argument('--output', default='artifacts/local_evaluation/limit_audit.json')
    args = parser.parse_args()
    runtime = RecoveryRuntime('policy', args.checkpoint)
    original_step = mujoco.mj_step
    rows = []
    try:
        for seed in range(args.seed, args.seed + args.episodes):
            runtime.reset(seed)
            row = {'seed': seed, 'max_position_violation_rad': 0., 'max_speed_ratio': 0.,
                   'max_commanded_torque_ratio': 0., 'physics_steps_audited': 0}

            def audited_step(model, data, *a, **kw):
                original_step(model, data, *a, **kw)
                info = runtime.info
                q = data.qpos[info.qadr]
                error = np.maximum(info.lower - q, q - info.upper).clip(0)
                speed = np.abs(data.qvel[info.vadr]) / info.velocity
                if error.max() > row['max_position_violation_rad']:
                    row.update(max_position_violation_rad=float(error.max()),
                               position_joint=info.names[int(error.argmax())])
                if speed.max() > row['max_speed_ratio']:
                    row.update(max_speed_ratio=float(speed.max()),
                               speed_joint=info.names[int(speed.argmax())])
                row['max_commanded_torque_ratio'] = max(row['max_commanded_torque_ratio'], float((np.abs(data.ctrl) / info.effort).max()))
                row['physics_steps_audited'] += 1

            mujoco.mj_step = audited_step
            try:
                for _ in range(750):
                    state = runtime.step()
                    if state['success'] or state['invalid']:
                        break
            finally:
                mujoco.mj_step = original_step
            row.update(posture_recovery_pass=state['success'], sim_time=state['sim_time'],
                       within_urdf_state_bounds=(row['max_position_violation_rad'] <= 1e-6 and row['max_speed_ratio'] <= 1.000001))
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        mujoco.mj_step = original_step
        runtime.close()
    result = {'checkpoint': args.checkpoint,
              'actor_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
              'sampling': 'Every 2ms physics step during the timed episode, excluding the initial settling interval.',
              'numeric_comparison_tolerance': {'position_rad': 1e-6, 'speed_ratio': 1e-6},
              'interpretation': 'Position-target and commanded torque bounds do not guarantee trajectory state bounds. Active soft joint constraints and torque-based speed limiting still permit transient state excursions. This audit is separate from the earlier posture-only recovery pass.',
              'episodes': len(rows), 'posture_recovery_passes': sum(r['posture_recovery_pass'] for r in rows),
              'within_urdf_state_bounds': sum(r['within_urdf_state_bounds'] for r in rows), 'results': rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
