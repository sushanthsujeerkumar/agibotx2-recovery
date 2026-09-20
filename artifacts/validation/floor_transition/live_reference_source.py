import sys,time,json
sys.path.insert(0,'scripts')
from screen_supported_lowering import target
from x2_recovery.runtime import RecoveryRuntime
from x2_recovery.physics import deep_crouch_target
import numpy as np
r=RecoveryRuntime(render=True,assess_stance=True,physics_profile='guarded_v2',reset_mode='deep_crouch',seed=9101)
r.display_label='SCRIPTED SUPPORT REFERENCE - NOT PPO OR FLOOR RECOVERY'
times=[0.,4.,6.,10.,13.,16.];poses=[deep_crouch_target(r.info),target(r.info,2.32,.9),target(r.info,2.32,.9),deep_crouch_target(r.info),r.info.nominal,r.info.nominal]
def desired():
 t=float(r.data.time);i=min(max(np.searchsorted(times,t,side='right')-1,0),len(times)-2);u=np.clip((t-times[i])/(times[i+1]-times[i]),0,1);u=u*u*(3-2*u);return poses[i]*(1-u)+poses[i+1]*u
r._scripted_target=desired
try:
 for _ in range(800):
  start=time.monotonic();s=r.step()
  if not r.viewer.is_running() or s['invalid'] or not s['trajectory_limits']['ok']:break
  time.sleep(max(0.,.02-(time.monotonic()-start)))
 print(json.dumps({k:v for k,v in s.items() if k not in ['joint_names','joint_positions']}))
finally:r.close()
