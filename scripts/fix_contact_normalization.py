#!/usr/bin/env python3
"""Create a separate PPO warm start with unit-scale categorical contact inputs."""
import argparse,hashlib,json
from pathlib import Path
import torch
from x2_recovery.train import export_actor
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--parent',default='artifacts/experiments/crouch_ppo/checkpoint.pt')
p.add_argument('--output',default='artifacts/experiments/deep_contact_normalization')
a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
if (out/'checkpoint.pt').exists():raise RuntimeError('Choose a fresh output directory')
c=torch.load(a.parent,map_location='cpu',weights_only=False)
s=c['actor_state_dict'];old=s['obs_normalizer._std'].clone()
# Observation106: joint offsets31, velocities31, gravity3, local velocity3,
# angular velocity3, height1, contact flags3, actual previous action31.
# EmpiricalNormalization uses std+.01. Unit denominator keeps binary contact
# changes bounded by1; retain means to preserve all demonstrated nominal inputs.
s['obs_normalizer._std'][...,72:75]=.99
s['obs_normalizer._var'][...,72:75]=.99**2
meta={'method':'contact normalization correction; no new optimizer steps',
      'parent':a.parent,'parent_sha256':hashlib.sha256(Path(a.parent).read_bytes()).hexdigest(),
      'contact_channels':[72,73,74],'old_std':old[...,72:75].tolist(),
      'new_effective_denominator':1.,'normalizer_eps':.01,
      'preserved':'actor weights, other observation statistics, physical dynamics and evaluation criteria',
      'scope':'separate candidate; select only after physical evaluations'}
c['initialization']=meta;c['warmstart_only']=True
for key in ['iter','completed_iterations','environment_steps','training_time_s','wall_time_s']:c[key]=0
for key in ['optimizer_state_dict','environment_state','rng_state']:c.pop(key,None)
c['train_cfg']['freeze_actor_normalization']=True
c['environment_cfg'].update(reset_mode='deep_crouch',reset='training_only_deep_crouch')
torch.save(c,out/'checkpoint.pt');export_actor(out/'checkpoint.pt',out/'actor.pt')
meta['actor_sha256']=hashlib.sha256((out/'actor.pt').read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
