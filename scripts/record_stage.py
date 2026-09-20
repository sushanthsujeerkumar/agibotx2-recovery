#!/usr/bin/env python3
"""Record an explicitly named start condition using only a frozen learned actor."""
import argparse, hashlib, json
from pathlib import Path
import imageio.v2 as imageio
import mujoco
from x2_recovery.runtime import RecoveryRuntime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--start',choices=['supine','crouch','balance'],required=True)
    p.add_argument('--seed',type=int,default=3001)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.seconds < .02: p.error('seconds must be at least one control step (.02)')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    runtime=RecoveryRuntime('policy',a.checkpoint,seed=a.seed,assess_stance=True,
                            physics_profile='guarded_v2',reset_mode=a.start)
    renderer=mujoco.Renderer(runtime.model,height=480,width=640)
    camera=mujoco.MjvCamera();camera.lookat[:]=[0,0,.65];camera.distance=2.7;camera.azimuth=135;camera.elevation=-15
    options=mujoco.MjvOption();options.geomgroup[3]=0
    history=[];initial_height=float(runtime.data.qpos[2])
    try:
        with imageio.get_writer(out/'episode.mp4',fps=25) as writer:
            for step in range(round(a.seconds/.02)):
                s=runtime.step()
                if step%2==0:
                    renderer.update_scene(runtime.data,camera=camera,scene_option=options)
                    frame=renderer.render();writer.append_data(frame)
                    if step==0:imageio.imwrite(out/'start.png',frame)
                if step%5==0:history.append({k:v for k,v in s.items() if k not in ['joint_names','joint_positions']})
                if s['invalid'] or not s['trajectory_limits']['ok']:break
            renderer.update_scene(runtime.data,camera=camera,scene_option=options)
            imageio.imwrite(out/'end.png',renderer.render())
        result={'task':f'{a.start}-start learned-policy rollout; only supine starts count as ground recovery',
                'checkpoint':a.checkpoint,'actor_sha256':hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),
                'seed':a.seed,'initial_height':initial_height,
                'completed_duration':step+1 == round(a.seconds/.02),
                'validated_clean_stage_pass':bool(step+1 == round(a.seconds/.02) and s['clean_stance_success'] and s['trajectory_limits']['ok'] and not s['invalid']),
                'monitoring':{'joint_limits_hz':1000,'posture_and_contacts_hz':50},
                'result':{k:v for k,v in s.items() if k not in ['joint_names','joint_positions']},'trajectory':history}
        (out/'episode.json').write_text(json.dumps(result,indent=2)+'\n')
    finally:renderer.close();runtime.close()

if __name__=='__main__':main()
