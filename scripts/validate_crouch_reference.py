"""Exploratory one-seed scripted crouch reference; not a recovery controller."""
import mujoco, numpy as np,json
from x2_recovery.common import ModelInfo,ground_forces,success_conditions
from x2_recovery.physics import reset_balance,TrajectoryLimits
from x2_recovery.stance import stance_metrics
rows=[]
for bend in [.6,1.,1.4,1.7]:
 i=ModelInfo(physics_profile='guarded_v2');m=i.model;d=mujoco.MjData(m);reset_balance(i,d,2001);audit=TrajectoryLimits(i)
 low=i.nominal.copy()
 for side in ['left','right']:
  for name,val in [('hip_pitch',-bend/2),('knee',bend),('ankle_pitch',-bend/2)]:low[i.names.index(side+'_'+name+'_joint')]=val
 minheight=1.;max_hold=0.;hold=0.
 for step in range(700):
  t=step*.02
  # Three-second lowering, one-second hold, three-second return, seven-second balance.
  phase=min(t/3,1.) if t<4 else max(0.,1.-(t-4)/3)
  phase=phase*phase*(3-2*phase)
  target=i.nominal*(1-phase)+low*phase
  for _ in range(i.substeps):
   d.ctrl[:]=i.torque(d.qpos[i.qadr],d.qvel[i.vadr],target);mujoco.mj_step(m,d);audit.observe(d)
  minheight=min(minheight,float(d.qpos[2]))
  clean=all(success_conditions(i,d,ground_forces(i,d)).values()) and stance_metrics(i,d)['posture_ok']
  hold=hold+.02 if clean and t>7 else 0.;max_hold=max(max_hold,hold)
  if not audit.ok or d.qpos[2]<.25:break
 row=dict(knee_bend=bend,time_s=float(d.time),min_height=minheight,final_height=float(d.qpos[2]),clean_hold_after_return=max_hold,limits=audit.report())
 rows.append(row);print(json.dumps(row),flush=True)
open('artifacts/validation/physics_v2/crouch_screen.json','w').write(json.dumps(rows,indent=2)+'\n')
