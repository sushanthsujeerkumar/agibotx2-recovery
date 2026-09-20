#!/usr/bin/env python3
"""Physical hands-supported squat to kneeling and back; feasibility only."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from x2_recovery.common import ModelInfo,CONTROL_DT,ground_forces,success_conditions
from x2_recovery.physics import reset_deep_crouch,deep_crouch_target,TrajectoryLimits
from x2_recovery.stance import stance_metrics
from screen_supported_lowering import contacts,target
CANDIDATES=[(h,a,-1.2,-.2) for h in [-2.,-1.6,-1.2] for a in [-.3,0.,.3]]+[(h,.15,-1.4,-.8) for h in [-2.,-1.6,-1.2]]

def run(info,candidate,seed):
 hip,ankle,shoulder,elbow=candidate;deep=deep_crouch_target(info);hands=target(info,2.32,.9);kneel=hands.copy()
 for side in ['left','right']:
  for part,value in [('hip_pitch',hip),('knee',2.2),('ankle_pitch',ankle),('shoulder_pitch',shoulder),('elbow',elbow)]:kneel[info.names.index(side+'_'+part+'_joint')]=value
 if np.any(kneel<info.lower+.05) or np.any(kneel>info.upper-.05):raise ValueError('Candidate outside guarded targets')
 d=mujoco.MjData(info.model);reset_deep_crouch(info,d,seed);audit=TrajectoryLimits(info);audit.observe(d)
 times=[0.,4.,6.,10.,12.,16.,18.,22.,25.,28.];poses=[deep,hands,hands,kneel,kneel,hands,hands,deep,info.nominal,info.nominal]
 rows=[];lowest=None;hold=0.;contacts_seen=set();kneel_time=0.;reason='duration'
 for step in range(round(28/CONTROL_DT)):
  t=step*CONTROL_DT;k=min(max(np.searchsorted(times,t,side='right')-1,0),len(times)-2);u=np.clip((t-times[k])/(times[k+1]-times[k]),0.,1.);u=u*u*(3-2*u);command=poses[k]*(1-u)+poses[k+1]*u
  for _ in range(info.substeps):
   d.ctrl[:]=info.torque(d.qpos[info.qadr],d.qvel[info.vadr],command);mujoco.mj_step(info.model,d);audit.observe(d)
   if not audit.ok:reason='limit_violation';break
  c=contacts(info,d);contacts_seen.update(c);checks=success_conditions(info,d,ground_forces(info,d));stance=stance_metrics(info,d)
  clean=all(checks.values()) and stance['posture_ok'] and audit.ok;hold=hold+CONTROL_DT if clean else 0.
  if any('_hip_yaw_link' in n or '_knee_link' in n for n in c):kneel_time+=CONTROL_DT
  row={'time':float(d.time),'pelvis_height':float(d.qpos[2]),'torso_up':float(d.xmat[info.torso_id].reshape(3,3)[2,2]),'contacts':c,'linear_speed':float(np.linalg.norm(d.qvel[:3])),'angular_speed':float(np.linalg.norm(d.qvel[3:6]))}
  if lowest is None or row['pelvis_height']<lowest['pelvis_height']:lowest={**row,'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist()}
  if step%5==0:rows.append(row)
  if reason!='duration':break
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.qpos[2]<.05:reason='invalid_state';break
 return {'candidate':{'hip':hip,'knee':2.2,'ankle':ankle,'shoulder':shoulder,'elbow':elbow},'joint_target':dict(zip(info.names,kneel.tolist())),'seed':seed,'complete':abs(d.time-28)<1e-6,'return_clean_pass':bool(abs(d.time-28)<1e-6 and hold>=2.-1e-8 and audit.ok),'terminal_clean_hold':hold,'knee_region_contact_seconds':kneel_time,'minimum_state':lowest,'contacts_seen':sorted(contacts_seen),'final':row,'reason':reason,'limits':audit.report(),'trajectory':rows}

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/validation/floor_transition/kneeling_bridge');p.add_argument('--seed',type=int,default=9001);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists():raise RuntimeError('Choose new output')
 info=ModelInfo(physics_profile='guarded_v2');rows=[]
 for i,c in enumerate(CANDIDATES):
  r=run(info,c,a.seed);rows.append(r);(out/f'candidate_{i:02}.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['candidate','return_clean_pass','knee_region_contact_seconds','contacts_seen','reason','limits']}),flush=True)
 (out/'summary.json').write_text(json.dumps({'scope':'scripted intermediate bridge only; no learned or supine recovery','candidates':len(rows),'return_passes':sum(r['return_clean_pass'] for r in rows),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'joint_monitor_hz':1000,'contact_monitor_hz':50,'results':rows},indent=2)+'\n')
if __name__=='__main__':main()
