"""External vendor reference/policy compatibility probe, not our trained recovery."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnx
import torch
import yaml
from onnx.reference import ReferenceEvaluator

from x2_recovery.common import ModelInfo, CONTROL_DT, reset_cpu, ground_forces, success_conditions
from x2_recovery.physics import TrajectoryLimits
from x2_recovery.stance import stance_metrics

torch.set_num_threads(1)
HERE = Path(__file__).resolve().parent


class VendorGraph:
    def __init__(self):
        self.path = HERE/'policy_b_lie_up.onnx'
        graph = onnx.load(self.path, load_external_data=False)
        onnx.checker.check_model(graph)
        self.metadata = {p.key:p.value for p in graph.metadata_props}
        self.names = self.metadata['joint_names'].split(',')
        self.defaults = np.fromstring(self.metadata['default_joint_pos'], sep=',')
        self.scale = np.fromstring(self.metadata['action_scale'], sep=',')
        weights = {v.name:torch.from_numpy(onnx.numpy_helper.to_array(v).copy()) for v in graph.graph.initializer}
        self.weights = weights
        self.references = {}
        constants = {}
        for node in graph.graph.node:
            if node.op_type == 'Constant':
                constants[node.output[0]] = onnx.numpy_helper.to_array(next(a.t for a in node.attribute if a.name=='value'))
            elif node.op_type == 'Gather':
                self.references[node.output[0]] = constants[node.input[0]]
        # Numerical parity is checked independently against the ONNX interpreter.
        evaluator = ReferenceEvaluator(graph)
        rng = np.random.default_rng(104)
        errors = []
        for frame in [0,50,200,343]:
            obs = (weights['normalizer._mean'].numpy() +
                   weights['onnx::Div_50'].numpy()*rng.normal(size=(1,151))).astype(np.float32)
            oracle = evaluator.run(None, {'obs':obs, 'time_step':np.array([[frame]],np.float32)})
            errors.append(float(np.max(np.abs(self.action(obs)-oracle[0][0]))))
            for key, value in zip(list(self.references),oracle[1:]):
                assert np.array_equal(value[0],self.references[key][frame])
        self.parity_max_error = max(errors)
        assert self.parity_max_error < 1e-4

    def action(self, observation):
        with torch.inference_mode():
            x = torch.as_tensor(observation,dtype=torch.float32).reshape(1,151)
            x = (x-self.weights['normalizer._mean'])/self.weights['onnx::Div_50']
            for i in [0,2,4,6]:
                x = torch.nn.functional.linear(x,self.weights[f'actor.{i}.weight'],self.weights[f'actor.{i}.bias'])
                if i!=6:x=torch.nn.functional.elu(x)
            return x.numpy()[0].copy()


def contacts(info,data):
    output = {}
    force = np.zeros(6)
    for c in range(data.ncon):
        contact = data.contact[c]
        if contact.geom1 in info.floor_geoms:other=contact.geom2
        elif contact.geom2 in info.floor_geoms:other=contact.geom1
        else:continue
        mujoco.mj_contactForce(info.model,data,c,force)
        if force[0]>1:
            name=info.model.body(int(info.model.geom_bodyid[other])).name
            output[name]=output.get(name,0.)+float(force[0])
    return output


def run(graph,mode,timescale,seed,finish='blend'):
    info=ModelInfo(physics_profile='guarded_v2');data=mujoco.MjData(info.model)
    order=np.array([info.names.index(n) for n in graph.names]);qadr=info.qadr[order];vadr=info.vadr[order]
    reset_cpu(info,data,seed);initial=data.qpos[info.qadr].copy()
    config=yaml.safe_load((HERE/'ground_init_config.yaml').read_text());init=info.nominal.copy()
    for name,value in zip(config['BaseConfig']['action_seq'],config['ACConfig']['robot']['default_dof_pos']):init[info.names.index(name)]=value
    reference=graph.references['joint_pos'];velocity=graph.references['joint_vel']
    end=5.+(len(reference)-1)*CONTROL_DT*timescale;duration=15. if finish=='feedback' else end+6.
    monitor=TrajectoryLimits(info);monitor.observe(data);previous=np.zeros(29,np.float32)
    rows=[];states=[];raw_clip_count=0;max_clip=0.;maxheight=float(data.qpos[2]);hold=maxhold=0.;reason='duration';last=info.nominal.copy()
    for step in range(round(duration/CONTROL_DT)):
        t=step*CONTROL_DT
        if t<5:
            u=min(t/3.,1.);u=u*u*(3-2*u);command=(1-u)*initial+u*init;phase='prepare'
        elif t<end or finish=='feedback':
            frame=min(int((t-5.)/CONTROL_DT/timescale),len(reference)-1)
            command=info.nominal.copy();phase=mode
            if mode=='reference':command[order]=reference[frame]
            else:
                rotation=data.xmat[info.model.body('pelvis').id].reshape(3,3)
                obs=np.concatenate([reference[frame],velocity[frame],rotation.T@np.array([0.,0.,-1.]),
                                    data.qvel[3:6]*.25,data.qpos[qadr]-graph.defaults,
                                    data.qvel[vadr]*.05,previous]).astype(np.float32)
                previous=graph.action(obs)
                command[order]=graph.defaults+graph.scale*previous
            last=command.copy()
        else:
            u=min((t-end)/3.,1.);u=u*u*(3-2*u);command=(1-u)*last+u*info.nominal;phase='standing_blend_hold'
        bounded=np.clip(command,info.lower+.05,info.upper-.05)
        difference=float(np.max(np.abs(command-bounded)));max_clip=max(max_clip,difference);raw_clip_count+=difference>1e-9
        for _ in range(info.substeps):
            data.ctrl[:]=info.torque(data.qpos[info.qadr],data.qvel[info.vadr],bounded)
            mujoco.mj_step(info.model,data);monitor.observe(data)
            if not monitor.ok:reason='limit_fault';break
        valid=all(success_conditions(info,data,ground_forces(info,data)).values()) and stance_metrics(info,data)['posture_ok'] and monitor.ok
        hold=hold+CONTROL_DT if valid else 0.;maxhold=max(maxhold,hold);maxheight=max(maxheight,float(data.qpos[2]))
        if step%5==0:
            rows.append({'time':float(data.time),'phase':phase,'pelvis_height':float(data.qpos[2]),'torso_up':float(data.xmat[info.torso_id].reshape(3,3)[2,2]),'contacts':contacts(info,data),'target_adjustment_rad':difference})
            states.append(data.qpos.copy())
        if not monitor.ok:break
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():reason='nonfinite';break
    return {'scope':'External vendor compatibility diagnostic; not our PPO, not official runner reproduction',
            'mode':mode,'finish':finish,'seed':seed,'reference_timescale':timescale,'planned_duration':duration,
            'sim_time':float(data.time),'reason':reason,'standing_return_pass':bool(hold>=2 and reason=='duration' and monitor.ok),
            'terminal_hold':hold,'max_clean_hold':maxhold,'max_pelvis_height':maxheight,'limits':monitor.report(),
            'target_adjustment_steps':int(raw_clip_count),'max_target_adjustment_rad':max_clip,
            'final_checks':success_conditions(info,data,ground_forces(info,data)),'final_stance':stance_metrics(info,data),
            'state_edits_after_reset':False,'graph_parity_max_abs_error':graph.parity_max_error,
            'observation_contract':'reference joint position29 + reference velocity29 + pelvis gravity3 + pelvis local angular velocity3*0.25 + q29-default + dq29*0.05 + previous raw action29',
            'action_contract':'metadata default + metadata scale * raw action; head held nominal; guarded target envelope applied and counted',
            'policy_sha256':hashlib.sha256(graph.path.read_bytes()).hexdigest(),'rows':rows}, np.array(states)


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['reference','vendor_actor'],required=True);p.add_argument('--timescale',type=float,default=1.);p.add_argument('--finish',choices=['blend','feedback'],default='blend');p.add_argument('--seed',type=int,default=9201);p.add_argument('--output',required=True);args=p.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():raise ValueError('Choose fresh output')
    graph=VendorGraph();result,states=run(graph,args.mode,args.timescale,args.seed,args.finish)
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(out/'physical_states.npz',qpos=states)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','final_stance']},indent=2),flush=True)


if __name__=='__main__':main()
