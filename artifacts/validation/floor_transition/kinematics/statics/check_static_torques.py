#!/usr/bin/env python3
"""STATIC ONLY LP: gravity/contact/actuator equilibrium at existing IK poses.

This script performs no integration, policy training, or IK search. It models
rigid point contacts and excludes useful assistance from joint frictionloss.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import linprog

from x2_recovery.common import ModelInfo


HERE = Path(__file__).resolve().parent
KINEMATICS = HERE.parent


def solve(info, data, points, torque_bounds):
    m = info.model
    nf = 3*len(points)
    n = nf+m.nu+1
    peak = n-1
    eq = np.zeros((m.nv, n))
    for i, point in enumerate(points):
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jac(m, data, jacp, jacr, np.array(point['position_xyz_m']), point['body_id'])
        eq[:, 3*i:3*i+3] = jacp.T
    for i, adr in enumerate(info.vadr):
        eq[adr, nf+i] = m.actuator_gear[i, 0]
    rows = []
    for i, point in enumerate(points):
        # Inscribed diamond: |fx|+|fy| <= mu*fz, conservative versus circular cone.
        for sx, sy in ((-1,-1), (-1,1), (1,-1), (1,1)):
            row = np.zeros(n)
            row[3*i:3*i+3] = [sx, sy, -point['mu']]
            rows.append(row)
    for i, effort in enumerate(info.effort):
        for sign in (-1, 1):
            row = np.zeros(n)
            row[nf+i], row[peak] = sign, -effort
            rows.append(row)
    bounds = []
    for _ in points:
        bounds.extend([(None,None), (None,None), (0,None)])
    bounds.extend(list(map(tuple, torque_bounds)))
    bounds.append((0,None))
    objective = np.zeros(n)
    objective[peak] = 1
    rhs = data.qfrc_bias - data.qfrc_passive
    lp = linprog(objective, A_ub=np.stack(rows), b_ub=np.zeros(len(rows)),
                 A_eq=eq, b_eq=rhs, bounds=bounds, method='highs',
                 options={'dual_feasibility_tolerance': 1e-8, 'primal_feasibility_tolerance': 1e-8})
    result = {'feasible': bool(lp.success), 'status': int(lp.status), 'message': lp.message}
    if not lp.success:
        return result
    forces = lp.x[:nf].reshape(-1,3)
    torques = lp.x[nf:nf+m.nu]
    residual = eq@lp.x-rhs
    result.update({
        'minimum_peak_effort_ratio': float(lp.x[-1]),
        'max_generalized_equilibrium_residual': float(np.max(np.abs(residual))),
        'base_force_balance_residual_N': residual[:3].tolist(),
        'base_moment_balance_residual_Nm': residual[3:6].tolist(),
        'joint_torques_Nm': dict(zip(info.names, map(float,torques))),
        'joint_effort_ratios': dict(zip(info.names, map(float,np.abs(torques)/info.effort))),
        'contact_forces': [dict(p, force_world_xyz_N=f.tolist(),
                               friction_l1_ratio=float((abs(f[0])+abs(f[1]))/(p['mu']*f[2])) if f[2]>1e-8 else None)
                           for p,f in zip(points,forces)],
        'normal_force_sum_N': float(np.sum(forces[:,2])),
    })
    return result


def run(name):
    info = ModelInfo(physics_profile='guarded_v2')
    m = info.model
    data = mujoco.MjData(m)
    posepath = KINEMATICS/f'{name}.json'
    pose = json.loads(posepath.read_text())
    data.qpos[:] = pose['qpos']
    data.qvel[:] = 0
    mujoco.mj_forward(m,data)
    assert np.all(m.actuator_gear[:,0] == 1) and np.all(m.actuator_gear[:,1:] == 0)
    points = []
    floor = m.geom('floor').id
    for contact in pose['proposed_contacts']:
        geom = m.geom(contact['geom']).id
        for position in contact['lowest_points_xyz_m']:
            assert abs(position[2]) < .001
            points.append({'geom': contact['geom'], 'body_id': int(m.geom_bodyid[geom]),
                           'position_xyz_m': [position[0],position[1],0.],
                           'original_height_error_m': position[2],
                           'mu': float(min(m.geom_friction[geom,0],m.geom_friction[floor,0]))})
    effort_bounds = np.column_stack([-info.effort,info.effort])
    q = data.qpos[info.qadr]
    zero = np.zeros(m.nu)
    servo_low = info.torque(q,zero,info.lower+.05)
    servo_high = info.torque(q,zero,info.upper-.05)
    published = solve(info,data,points,effort_bounds)
    guarded = solve(info,data,points,np.column_stack([servo_low,servo_high]))
    zero_torque = solve(info,data,points,np.column_stack([zero,zero]))
    report = {
        'label': 'STATIC ONLY; no physical transition, simulation rollout, or recovery success',
        'pose': name, 'input_pose_sha256': hashlib.sha256(posepath.read_bytes()).hexdigest(),
        'mass_kg': float(sum(m.body_mass)), 'gravity_m_s2': m.opt.gravity.tolist(),
        'num_contact_points': len(points), 'qfrc_bias': data.qfrc_bias.tolist(),
        'qfrc_passive': data.qfrc_passive.tolist(),
        'equation': 'sum(J_point.T @ f_point) + actuator_torque = qfrc_bias - qfrc_passive, all 37 floating-base degrees',
        'friction': '|fx| + |fy| <= 0.8 fz; fz >= 0; no contact torsion or rolling moments',
        'published_effort': published, 'guarded_servo_attainable': guarded,
        'zero_torque_at_exact_pose_target': zero_torque,
        'servo_attainable_torque_bounds_Nm': {n:[float(l),float(h)] for n,l,h in zip(info.names,servo_low,servo_high)},
        'assumptions': [
            'Contacts treated as rigid stationary point supports at proposed lowest primitive points.',
            'Submicrometre gaps projected to plane; contact compliance/penetration not solved.',
            'No joint frictionloss assistance included; qfrc_passive at zero velocity is included.',
            'No contact pressure limits or structural bounds beyond model actuator effort.',
            'Inscribed friction diamond is more conservative than a circular Coulomb cone.',
            'LP feasibility does not imply stability, reachability, robustness, or that the passive contact solver selects these forces.',
            'Inverse servo offsets apply only at this exact pose and zero joint velocities.',
        ]
    }
    if guarded['feasible']:
        torque = np.array([guarded['joint_torques_Nm'][n] for n in info.names])
        damping = np.maximum(info.kd,info.effort/(.25*info.velocity))
        stiffness = damping*(info.kp/info.kd)
        delta = torque/stiffness
        target = q+delta
        actual = info.torque(q,zero,target)
        report['gravity_offset_targets'] = {
            'equation': 'target = q + tau / (max(kd, effort/(0.25*velocity)) * (kp/kd)); checked through exact guarded_torque',
            'effective_small_error_stiffness_Nm_rad': dict(zip(info.names,map(float,stiffness))),
            'offset_rad': dict(zip(info.names,map(float,delta))),
            'target_rad': dict(zip(info.names,map(float,target))),
            'max_abs_offset_rad': float(np.max(np.abs(delta))),
            'max_servo_torque_error_Nm': float(np.max(np.abs(actual-torque))),
            'minimum_target_guarded_margin_rad': float(np.min(np.minimum(target-info.lower-.05,info.upper-.05-target))),
            'within_guarded_target_bounds': bool(np.all(target>=info.lower+.05-1e-10) and np.all(target<=info.upper-.05+1e-10)),
            'unoffset_torque_Nm': info.torque(q,zero,q).tolist(),
        }
    if not published['feasible']:
        report['unbounded_actuator_minimum_effort_scale'] = solve(info,data,points,[(None,None)]*m.nu)
    (HERE/f'{name}.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pose':name,'published_feasible':published['feasible'],
                      'guarded_feasible':guarded['feasible'],
                      'zero_torque_feasible':zero_torque['feasible'],
                      'peak_ratio':guarded.get('minimum_peak_effort_ratio'),
                      'offset':report.get('gravity_offset_targets',{}).get('max_abs_offset_rad')}),flush=True)
    return report


def main():
    start = time.monotonic()
    reports = [run(n) for n in ('hands_knee_region','half_knee_region','seat_feet_hands')]
    (HERE/'summary.json').write_text(json.dumps({
        'label':'STATIC ONLY', 'physics_steps':0, 'elapsed_seconds':time.monotonic()-start,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'results':reports,
    },indent=2)+'\n')


if __name__=='__main__':
    main()
