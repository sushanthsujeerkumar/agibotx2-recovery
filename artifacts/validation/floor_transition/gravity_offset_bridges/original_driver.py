from pathlib import Path
import sys,json,numpy as np
sys.path.insert(0,'scripts')
from validate_ik_bridges import episode
from x2_recovery.common import ModelInfo
out=Path('artifacts/validation/floor_transition/gravity_offset_bridges');out.mkdir(parents=True,exist_ok=True);info=ModelInfo(physics_profile='guarded_v2');rows=[]
for name,start in [('half_knee_region','deep_crouch'),('seat_feet_hands','supine')]:
 r,states=episode(info,name,start,gravity_offsets=True);rows.append(r);label=start+'_'+name;(out/(label+'.json')).write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(out/(label+'_states.npz'),qpos=states);print(json.dumps({k:r[k] for k in ['name','start','standing_return_pass','reason','limits']}),flush=True)
(out/'summary.json').write_text(json.dumps({'scope':'Two physical checks of static gravity-offset targets; no learned recovery','results':rows},indent=2)+'\n')
