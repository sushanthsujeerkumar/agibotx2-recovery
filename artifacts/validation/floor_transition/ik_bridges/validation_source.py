#!/usr/bin/env python3
"""Replay offline joint proposals through physical dynamics; never inject IK base states."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from x2_recovery.common import ModelInfo,CONTROL_DT,ground_forces,success_conditions,reset_cpu
from x2_recovery.physics import reset_deep_crouch,deep_crouch_target,TrajectoryLimits
from x2_recovery.stance import stance_metrics
from screen_supported_lowering import target as low_target,contacts
from screen_supine_support import candidates,target as floor_target


def episode(info,name,start_mode,seed=9001):
 plan=Path('artifacts/validation/floor_transition/kinematics')/(name+'.json');payload=json.loads(plan.read_text());ik=np.array([payload['joint_targets_rad'][n] for n in info.names]);hands=low_target(info,2.32,.9);deep=deep_crouch_target(info)
 if np.any(ik<info.lower+.05) or np.any(ik>info.upper-.05):raise ValueError('Out-of-bound IK command')
 d=mujoco.MjData(info.model)
 if start_mode=='deep_crouch':
  reset_deep_crouch(info,d,seed)
  times=[0.,4.,6.,10.,12.,16.,18.,22.,25.,28.];poses=[deep,hands,hands,ik,ik,hands,hands,deep,info.nominal,info.nominal]
 else:
  reset_cpu(info,d,seed);initial=d.qpos[info.qadr].copy();_,press1,press2=candidates()[10]
  times=[0.,3.,5.,8.,12.,16.,18.,22.,24.,28.,31.,34.];poses=[initial,floor_target(info,press1),floor_target(info,press1),floor_target(info,press2),floor_target(info,press2),ik,ik,hands,hands,deep,info.nominal,info.nominal]
 audit=TrajectoryLimits(info);audit.observe(d);rows=[];states=[];hold=maxhold=0.;reason='duration';contact_set=set()
 for step in range(round(times[-1]/CONTROL_DT)):
  t=step*CONTROL_DT;k=min(max(np.searchsorted(times,t,side='right')-1,0),len(times)-2);u=np.clip((t-times[k])/(times[k+1]-times[k]),0.,1.);u=u*u*(3-2*u);command=poses[k]*(1-u)+poses[k+1]*u
  for _ in range(info.substeps):
   d.ctrl[:]=info.torque(d.qpos[info.qadr],d.qvel[info.vadr],command);mujoco.mj_step(info.model,d);audit.observe(d)
   if not audit.ok:reason='limit_fault';break
  c=contacts(info,d);contact_set.update(c);checks=success_conditions(info,d,ground_forces(info,d));stance=stance_metrics(info,d);clean=all(checks.values()) and stance['posture_ok'] and audit.ok;hold=hold+CONTROL_DT if clean else 0.;maxhold=max(maxhold,hold)
  if step%5==0:
   rows.append({'time':float(d.time),'pelvis_height':float(d.qpos[2]),'torso_up':float(d.xmat[info.torso_id].reshape(3,3)[2,2]),'contacts':c,'linear_speed':float(np.linalg.norm(d.qvel[:3])),'angular_speed':float(np.linalg.norm(d.qvel[3:6])),'checks':checks,'stance':stance})
   states.append(d.qpos.copy())
  if reason!='duration':break
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.qpos[2]<.05:reason='invalid_state';break
 return {'scope':'Scripted physical waypoint test; long reference timing, not learned or standard15s assessment','name':name,'start':start_mode,'seed':seed,'times':times,'command_poses':np.array(poses).tolist(),'ik_file':str(plan),'ik_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'complete':abs(d.time-times[-1])<1e-6,'standing_return_pass':bool(abs(d.time-times[-1])<1e-6 and hold>=2.-1e-8 and audit.ok),'max_clean_hold':maxhold,'terminal_hold':hold,'reason':reason,'limits':audit.report(),'contacts_seen':sorted(contact_set),'trajectory':rows},np.array(states)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/validation/floor_transition/ik_bridges');a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists():raise RuntimeError('Choose fresh output')
 info=ModelInfo(physics_profile='guarded_v2');rows=[]
 for start in ['deep_crouch','supine']:
  for name in ['half_knee_region','hands_knee_region','seat_feet_hands']:
   r,states=episode(info,name,start);rows.append(r);label=start+'_'+name;(out/(label+'.json')).write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(out/(label+'_states.npz'),qpos=states)
   print(json.dumps({k:r[k] for k in ['name','start','complete','standing_return_pass','max_clean_hold','reason','limits','contacts_seen']}),flush=True)
 (out/'summary.json').write_text(json.dumps({'scope':'Six physical waypoint screens, not new learned recovery','results':rows,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
if __name__=='__main__':main()
