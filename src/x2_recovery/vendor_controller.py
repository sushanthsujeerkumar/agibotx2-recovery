"""Pinned external controller and its explicitly attributed local target adapter."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from .common import CONTROL_DT

EXPECTED_POLICY_SHA256 = "3faf3df8f9616448f9ae580e22c2fa28fcb25c1eb0c9a338a0b0c5b9a520522f"

class VendorGraph:
    def __init__(self, asset_directory):
        import onnx
        from onnx.reference import ReferenceEvaluator
        self.directory = Path(asset_directory)
        self.path = self.directory/'policy_b_lie_up.onnx'
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != EXPECTED_POLICY_SHA256:
            raise ValueError('External policy hash differs from the inspected version')
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



class Teacher:
    def __init__(self, graph, info, data, timescale=1.2):
        import yaml
        self.graph = graph
        self.order = np.array([info.names.index(n) for n in graph.names])
        self.initial = data.qpos[info.qadr].copy()
        self.init = info.nominal.copy()
        cfg = yaml.safe_load((graph.directory / 'ground_init_config.yaml').read_text())
        for name, value in zip(cfg['BaseConfig']['action_seq'], cfg['ACConfig']['robot']['default_dof_pos']):
            self.init[info.names.index(name)] = value
        self.previous = np.zeros(29, np.float32)
        self.timescale = timescale

    def target(self, info, data, t):
        if t < 5:
            u = min(t / 3., 1.)
            u = u * u * (3 - 2 * u)
            command = (1 - u) * self.initial + u * self.init
        else:
            g = self.graph
            frame = min(int((t - 5) / CONTROL_DT / self.timescale), len(g.references['joint_pos']) - 1)
            rotation = data.xmat[info.model.body('pelvis').id].reshape(3, 3)
            obs = np.concatenate([g.references['joint_pos'][frame],
                                  g.references['joint_vel'][frame] / self.timescale,
                                  rotation.T @ np.array([0., 0., -1.]), data.qvel[3:6] * .25,
                                  data.qpos[info.qadr[self.order]] - g.defaults,
                                  data.qvel[info.vadr[self.order]] * .05, self.previous]).astype(np.float32)
            self.previous = g.action(obs)
            command = info.nominal.copy()
            command[self.order] = g.defaults + g.scale * self.previous
        return np.clip(command, info.lower + .05, info.upper - .05)
