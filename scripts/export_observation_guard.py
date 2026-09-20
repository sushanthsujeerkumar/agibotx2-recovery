import hashlib,json
from pathlib import Path
import torch
class Guard(torch.nn.Module):
    def __init__(self,parent,mean,std):
        super().__init__();self.parent=parent
        self.register_buffer('mean',mean);self.register_buffer('scale',std+.01)
        # Keep measured gravity, base height and discrete contact flags unchanged.
        mask=torch.ones(106,dtype=torch.bool);mask[62:65]=False;mask[71:75]=False
        self.register_buffer('mask',mask)
    def forward(self,x):
        clamped=self.mean+((x-self.mean)/self.scale).clamp(-5.,5.)*self.scale
        return self.parent(torch.where(self.mask,clamped,x))
base=Path('artifacts/experiments/crouch_ppo')
out=Path('artifacts/experiments/deep_observation_guard');out.mkdir(parents=True,exist_ok=True)
ck=torch.load(base/'checkpoint.pt',map_location='cpu',weights_only=False)
p=torch.jit.load(str(base/'actor.pt')).eval()
g=Guard(p,ck['actor_state_dict']['obs_normalizer._mean'],ck['actor_state_dict']['obs_normalizer._std']).eval()
torch.jit.script(g).save(str(out/'actor.pt'))
(out/'manifest.json').write_text(json.dumps({'method':'experimental inference observation guard; no new training','parent_actor_sha256':hashlib.sha256((base/'actor.pt').read_bytes()).hexdigest(),'normalized_clip':5.,'normalizer_eps':.01,'excluded_channels':list(range(62,65))+list(range(71,75)),'note':'Clips selected policy inputs only; physical state, actions and independent evaluation are unchanged. Not selected pending physical evaluation.'},indent=2)+'\n')
