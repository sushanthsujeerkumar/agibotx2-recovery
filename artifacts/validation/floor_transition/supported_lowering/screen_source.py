#!/usr/bin/env python3
"""Bounded physical lower-and-return screen; scripted reference, not learned recovery."""
import argparse,hashlib,json
from pathlib import Path
import mujoco
import numpy as np
from x2_recovery.common import ModelInfo,CONTROL_DT,ground_forces,success_conditions
from x2_recovery.physics import reset_deep_crouch,deep_crouch_target,TrajectoryLimits
from x2_recovery.stance import stance_metrics
CANDIDATES=[(k,l) for k in [1.8,2.,2.2] for l in [.2,.4,.6]]+[(2.32,l) for l in [.6,.9,1.2]]

def target(info,knee,lean):
 q=deep_crouch_target(info)
 for side in ['left','right']:
  for part,v in [('hip_pitch',-knee+.7-lean),('knee',knee),('ankle_pitch',-.7),('shoulder_pitch',-lean),('elbow',-.12)]:q[info.names.index(f'{side}_{part}_joint')]=v
 if np.any(q<info.lower+.05) or np.any(q>info.upper-.05):raise ValueError('Target outside guarded limits')
 return q

def contacts(info,d):
 result={}
 for c in d.contact:
  if c.geom1!=0 and c.geom2!=0:continue
  other=c.geom2 if c.geom1==0 else c.geom1
  f=np.zeros(6);mujoco.mj_contactForce(info.model,d,c.id if hasattr(c,'id') else 0,f)
 # Use indices, since MjContact does not carry its contact-list index.
 result={}
 for i in range(d.ncon):
  c=d.contact[i]
  if c.geom1!=0 and c.geom2!=0:continue
  other=c.geom2 if c.geom1==0 else c.geom1
  force=np.zeros(6);mujoco.mj_contactForce(info.model,d,i,force)
  if force[0]>1.:
   body=info.model.body(int(info.model.geom_bodyid[other])).name
   result[body]=result.get(body,0.)+float(force[0])
 return result

def run(info,knee,lean,seed):
 d=mujoco.MjData(info.model);reset_deep_crouch(info,d,seed);monitor=TrajectoryLimits(info);monitor.observe(d)
 low=target(info,knee,lean);start=deep_crouch_target(info)
 times=[0.,4.,6.,10.,13.,16.];poses=[start,low,low,start,info.nominal,info.nominal]
 min_height=float(d.qpos[2]);min_snapshot={};contact_set=set();hold=maxhold=0.;rows=[];reason='duration'
 for step in range(round(16/CONTROL_DT)):
  t=step*CONTROL_DT;k=min(max(np.searchsorted(times,t,side='right')-1,0),len(times)-2)
  u=np.clip((t-times[k])/(times[k+1]-times[k]),0.,1.);u=u*u*(3-2*u);command=poses[k]*(1-u)+poses[k+1]*u
  for _ in range(info.substeps):
   d.ctrl[:]=info.torque(d.qpos[info.qadr],d.qvel[info.vadr],command);mujoco.mj_step(info.model,d);monitor.observe(d)
   if not monitor.ok:reason='limit_violation';break
  support=contacts(info,d);contact_set.update(support)
  checks=success_conditions(info,d,ground_forces(info,d));stance=stance_metrics(info,d)
  clean=all(checks.values()) and stance['posture_ok'] and monitor.ok
  hold=hold+CONTROL_DT if clean else 0.;maxhold=max(maxhold,hold)
  row={'time':float(d.time),'pelvis_height':float(d.qpos[2]),'torso_up':float(d.xmat[info.torso_id].reshape(3,3)[2,2]),'contacts':support,'linear_speed':float(np.linalg.norm(d.qvel[:3])),'angular_speed':float(np.linalg.norm(d.qvel[3:6]))}
  if d.qpos[2]<min_height:min_height=float(d.qpos[2]);min_snapshot={**row,'qpos':d.qpos.tolist(),'qvel':d.qvel.tolist()}
  if step%5==0:rows.append(row)
  if reason!='duration':break
  if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all() or d.qpos[2]<.05:reason='invalid_state';break
 complete=abs(d.time-16)<1e-6
 return {'seed':seed,'knee':knee,'requested_lean':lean,'joint_targets':dict(zip(info.names,low.tolist())),'minimum_height':min_height,'minimum_state':min_snapshot,'ground_contact_bodies':sorted(contact_set),'complete':complete,'sim_time':float(d.time),'reason':reason,'return_clean_pass':bool(complete and hold>=2.-1e-8 and monitor.ok),'terminal_clean_hold':hold,'max_clean_hold':maxhold,'limits':monitor.report(),'final':row,'trajectory':rows}

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='artifacts/validation/floor_transition/supported_lowering');p.add_argument('--seed',type=int,default=9001);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 if (out/'summary.json').exists():raise RuntimeError('Choose fresh output')
 info=ModelInfo(physics_profile='guarded_v2');rows=[]
 for i,(k,l) in enumerate(CANDIDATES):
  try:r=run(info,k,l,a.seed)
  except ValueError as e:r={'knee':k,'requested_lean':l,'rejected_target':str(e),'return_clean_pass':False}
  rows.append(r);(out/f'candidate_{i:02}.json').write_text(json.dumps(r,indent=2)+'\n')
  print(json.dumps({k:r.get(k) for k in ['knee','requested_lean','minimum_height','return_clean_pass','reason','ground_contact_bodies','limits']}),flush=True)
 result={'scope':'Scripted lower-and-return feasibility, not learned or supine recovery','candidates':len(rows),'return_passes':sum(r['return_clean_pass'] for r in rows),'state_edits_after_reset':False,'monitoring':{'joint_hz':1000,'contacts_posture_hz':50},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'results':rows}
 (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
