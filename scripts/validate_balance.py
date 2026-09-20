#!/usr/bin/env python3
"""Standing-start validation only. These episodes are never recovery results."""
import argparse
import hashlib
import json
from pathlib import Path
import mujoco
import numpy as np
from x2_recovery.common import ModelInfo, observation_numpy, sensor_forces, ground_forces, success_conditions
from x2_recovery.physics import reset_balance, reset_crouch, TrajectoryLimits
from x2_recovery.stance import stance_metrics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint')
    p.add_argument('--start',choices=['balance','crouch'],default='balance')
    p.add_argument('--episodes',type=int,default=5)
    p.add_argument('--seed',type=int,default=2001)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--output',default='artifacts/validation/physics_v2/standing_validation.json')
    args=p.parse_args()
    if args.episodes<1 or args.seconds<=0: p.error('episodes and seconds must be positive')
    info=ModelInfo(physics_profile='guarded_v2');m=info.model;d=mujoco.MjData(m)
    policy=None
    if args.checkpoint:
        import torch
        torch.set_num_threads(2)
        policy=torch.jit.load(args.checkpoint,map_location='cpu').eval()
    rows=[]
    for seed in range(args.seed,args.seed+args.episodes):
        (reset_balance if args.start=='balance' else reset_crouch)(info,d,seed)
        initial_height=float(d.qpos[2])
        monitor=TrajectoryLimits(info);monitor.observe(d)
        hold=0.;max_hold=0.;previous=np.zeros(m.nu);ever_passed=False
        for step in range(round(args.seconds/.02)):
            if policy is None:
                action=np.zeros(m.nu)
            else:
                obs=observation_numpy(info,d,previous,sensor_forces(info,d))
                with torch.inference_mode():
                    action=policy(torch.from_numpy(obs)[None]).squeeze(0).numpy()
                if action.shape != (m.nu,) or not np.isfinite(action).all():
                    raise RuntimeError('Policy returned invalid actions')
                action=action.clip(-1,1)
            target=info.targets(action)
            for _ in range(info.substeps):
                d.ctrl[:]=info.torque(d.qpos[info.qadr],d.qvel[info.vadr],target)
                mujoco.mj_step(m,d);monitor.observe(d)
            previous=action.copy()
            checks=success_conditions(info,d,ground_forces(info,d));stance=stance_metrics(info,d)
            clean=all(checks.values()) and stance['posture_ok'] and monitor.ok
            hold=hold+.02 if clean else 0.;max_hold=max(max_hold,hold)
            ever_passed |= hold>=2.-1e-8
            if not monitor.ok or not np.isfinite(d.qpos).all() or d.qpos[2]<.3:break
        completed=step+1 == round(args.seconds/.02)
        row=dict(seed=seed,initial_height=initial_height,sim_time=float(d.time),clean_balance_pass=bool(ever_passed and monitor.ok),
                 completed_duration=completed,sustained_clean_pass=bool(ever_passed and monitor.ok and clean and completed),
                 clean_at_end=bool(clean),max_clean_hold_s=max_hold,limits=monitor.report(),final_checks=checks,stance=stance)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k not in ['stance','final_checks']}),flush=True)
    result=dict(task=f'{args.start}-start balance; NOT supine recovery',physics_profile='guarded_v2',
                checkpoint=args.checkpoint,actor_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest() if args.checkpoint else None,
                episodes=len(rows),clean_balance_passes=sum(r['clean_balance_pass'] for r in rows),
                trajectory_limit_passes=sum(r['limits']['ok'] for r in rows),results=rows)
    result.update(sustained_clean_passes=sum(r['sustained_clean_pass'] for r in rows),
                  monitoring={'joint_limits_hz':1000,'posture_and_contacts_hz':50},
                  pass_definition='clean_balance_pass: attained two-second clean hold within limits; sustained_clean_pass additionally requires full duration and clean final state')
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
