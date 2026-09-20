#!/usr/bin/env python3
"""Fit actor actions to audited scripted demonstrations, separately from PPO."""
import argparse, copy, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from x2_recovery.train import export_actor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--parent',default='artifacts/experiments/balance_prior_v2/checkpoint.pt')
    p.add_argument('--output',default='artifacts/experiments/crouch_teacher')
    p.add_argument('--steps',type=int,default=2500)
    p.add_argument('--seed',type=int,default=4)
    p.add_argument('--validation-seeds',default='',help='Comma-separated held-out demonstration seeds; never physical evaluation seeds')
    p.add_argument('--observation-noise',type=float,default=0.,help='Raw continuous-observation noise std; contact indicators remain unchanged')
    args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'checkpoint.pt').exists():raise RuntimeError('Choose a fresh output directory')
    data=np.load(args.dataset)
    print('Dataset keys:',list(data.files),flush=True)
    x=torch.tensor(data['observations'],dtype=torch.float32)
    y=torch.tensor(data['actions'],dtype=torch.float32)
    seeds=np.asarray(data['seeds'])[np.asarray(data['episode_ids'])]
    distinct=np.unique(seeds)
    validation_seeds=np.asarray([int(s) for s in args.validation_seeds.split(',')]) if args.validation_seeds else distinct[-4:]
    valid_mask=torch.tensor(np.isin(seeds,validation_seeds))
    assert len(distinct)>=8 and x.shape[1]==106 and y.shape==(len(x),31)
    assert torch.isfinite(x).all() and torch.isfinite(y).all() and y.abs().max()<=1.
    if not valid_mask.any() or valid_mask.all():raise ValueError('Both training and validation seeds are required')
    checkpoint=torch.load(args.parent,map_location='cpu',weights_only=False)
    cfg=copy.deepcopy(checkpoint['train_cfg']);acfg=copy.deepcopy(cfg['actor']);acfg.pop('class_name')
    model=MLPModel(TensorDict({'actor':x,'critic':x},batch_size=[len(x)]),cfg['obs_groups'],'actor',31,**acfg)
    model.load_state_dict(checkpoint['actor_state_dict'])
    # Refit the observation statistics to the training demonstration seeds only.
    from rsl_rl.modules import EmpiricalNormalization
    model.obs_normalizer=EmpiricalNormalization(106)
    model.obs_normalizer.update(x[~valid_mask]);model.obs_normalizer.until=0
    opt=torch.optim.Adam(model.mlp.parameters(),lr=3e-4)
    train_ids=torch.where(~valid_mask)[0]
    transition_ids=train_ids[y[train_ids].abs().amax(-1)>.01]
    start=time.monotonic();logs=[];best=float('inf');best_state=None;best_step=0
    for step in range(args.steps):
        # Equal emphasis on transition and the entire trajectory, including standing.
        ids=torch.cat([train_ids[torch.randint(len(train_ids),(256,))],transition_ids[torch.randint(len(transition_ids),(256,))]])
        noisy=x[ids].clone()
        if args.observation_noise:
            noise=torch.randn_like(noisy)*args.observation_noise;noise[:,72:75]=0.
            noisy+=noise
        batch=TensorDict({'actor':noisy},batch_size=[len(ids)])
        prediction=model(batch)
        loss=(prediction-y[ids]).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.mlp.parameters(),1.);opt.step()
        if (step+1)%100==0 or step==0:
            with torch.no_grad():
                pred=model(TensorDict({'actor':x[valid_mask]},batch_size=[int(valid_mask.sum())]))
                val=(pred-y[valid_mask]).square().mean().item()
            row=dict(step=step+1,training_mse=float(loss.detach()),validation_mse=val,wall_seconds=time.monotonic()-start)
            logs.append(row);print(json.dumps(row),flush=True)
            if val<best:best=val;best_state=copy.deepcopy(model.state_dict());best_step=step+1
    model.load_state_dict(best_state)
    provenance=dict(method='behavior_cloning_from_audited_scripted_crouch_reference',
                    dataset=args.dataset,dataset_sha256=hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                    parent=args.parent,parent_sha256=hashlib.sha256(Path(args.parent).read_bytes()).hexdigest(),
                    training_seeds=np.setdiff1d(distinct,validation_seeds).tolist(),validation_seeds=validation_seeds.tolist(),
                    observation_noise_std=args.observation_noise,
                    gradient_steps=args.steps,best_step=best_step,best_validation_mse=best,wall_seconds=time.monotonic()-start,
                    note='Supervised warm start only; zero new PPO iterations. Simulation rollout, not MSE, determines success.')
    checkpoint['actor_state_dict']=model.state_dict()
    checkpoint['initialization']=provenance
    for key in ['iter','completed_iterations','environment_steps','training_time_s','wall_time_s']:checkpoint[key]=0
    checkpoint['environment_cfg'].update(reset_mode='crouch',reset='training_only_crouch')
    checkpoint['warmstart_only']=True
    for key in ['optimizer_state_dict','environment_state','rng_state']:
        checkpoint.pop(key,None)
    # PPO continuation must use initialize-from: optimizer/critic were not supervised.
    torch.save(checkpoint,out/'checkpoint.pt');export_actor(out/'checkpoint.pt',out/'actor.pt')
    (out/'fit.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'fit_progress.json').write_text(json.dumps(logs,indent=2)+'\n')

if __name__=='__main__':main()
