#!/usr/bin/env python3
"""One bounded STATIC/KINEMATIC ONLY continuation from attained squat to kneel.

Twelve waypoints maximum; fixed left-foot/both-hand support locations. Never
integrates physics or injects these configurations into the runtime.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation, Slerp

from plan_support_posture import Planner


ROOT = Path(__file__).resolve().parents[1]
KIN = ROOT/'artifacts/validation/floor_transition/kinematics'
OUT = KIN/'path'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sourcepath = ROOT/'artifacts/validation/floor_transition/actual_hand_support.json'
    source = json.loads(sourcepath.read_text())
    endpoint = json.loads((KIN/'half_knee_region.json').read_text())
    p = Planner()
    m, d, info = p.m, p.d, p.info
    d.qpos[:] = source['qpos']
    mujoco.mj_forward(m,d)
    q0 = d.qpos.copy()
    x0 = np.r_[q0[:3],Rotation.from_quat(q0[[4,5,6,3]]).as_rotvec(),q0[info.qadr]]
    active = [p.feet[0]]+p.hands
    anchors = []
    for geom in active:
        body = int(m.geom_bodyid[geom])
        for point in p.bottom(geom)[1]:
            anchors.append({'geom':m.geom(geom).name, 'body_id':body,
                            'local_point':(d.xmat[body].reshape(3,3).T@(point-d.xpos[body])).tolist(),
                            'world_goal':[float(point[0]),float(point[1]),0.],
                            'original_z':float(point[2])})
    anchorxy = np.array([a['world_goal'][:2] for a in anchors])
    hull = ConvexHull(anchorxy)
    right_knee = p.knee_region[1]
    right_foot = p.feet[1]
    qend = np.array(endpoint['qpos'])
    xend = np.r_[qend[:3],Rotation.from_quat(qend[[4,5,6,3]]).as_rotvec(),qend[info.qadr]]
    # Translation aligns the endpoint's left foot to the actual support centroid.
    d.qpos[:] = qend
    mujoco.mj_forward(m,d)
    endfoot = np.mean(p.bottom(p.feet[0])[1],axis=0)
    leftanchors = np.mean([a['world_goal'] for a in anchors if a['geom']==m.geom(p.feet[0]).name],axis=0)
    xend[:2] += leftanchors[:2]-endfoot[:2]
    bounds = np.column_stack((np.r_[x0[:2]-.45,.10,[-.5,-.4,-.5],info.lower+.05],
                              np.r_[x0[:2]+.45,.65,[.5,1.7,.5],info.upper-.05]))

    def set_pose(x):
        d.qpos[:3] = x[:3]
        quat = Rotation.from_rotvec(x[3:6]).as_quat()
        d.qpos[3:7] = quat[[3,0,1,2]]
        d.qpos[info.qadr] = x[6:]
        d.qvel[:] = 0
        mujoco.mj_forward(m,d)

    def contact_positions():
        return np.array([d.xpos[a['body_id']]+d.xmat[a['body_id']].reshape(3,3)@a['local_point'] for a in anchors])

    def residual(x, phase, previous):
        set_pose(x)
        actual = contact_positions()
        target = np.array([a['world_goal'] for a in anchors])
        com = d.subtree_com[m.body('pelvis').id]
        bottoms = {g:p.bottom(g)[0] for g in p.ids}
        res = list((150*(actual-target)).ravel())
        res.extend(150*min(0.,z) for z in bottoms.values())
        res.append(150*p.self_penetration())
        res.extend(80*np.maximum(hull.equations[:,:2]@com[:2]+hull.equations[:,2]+.015,0.))
        # A single continuation schedule transfers the right leg. No support slip.
        jointgoal = x0[6:]*(1-phase)+xend[6:]*phase
        for suffix in ('hip_pitch_joint','knee_joint','ankle_pitch_joint'):
            j = info.names.index('right_'+suffix)
            res.append(1.5*(x[6+j]-jointgoal[j]))
        if phase<1.:
            clearance = .015*np.sin(np.pi*phase)
            res.append(100*min(0.,bottoms[right_foot]-clearance))
        else:
            res.append(150*bottoms[right_knee])
        res.extend(.06*(x-(x0*(1-phase)+xend*phase)))
        res.extend(.03*(x-previous))
        return np.asarray(res)

    def report(x,phase,sol):
        set_pose(x)
        apos = contact_positions()
        error = apos-np.array([a['world_goal'] for a in anchors])
        com = d.subtree_com[m.body('pelvis').id]
        margin = float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]))
        allclear = {m.geom(g).name:p.bottom(g)[0] for g in p.ids}
        minfloor = min(allclear.values())
        selfpen = p.self_penetration()
        maxanchor = float(np.max(np.linalg.norm(error,axis=1)))
        kneegap = p.bottom(right_knee)[0]
        reasons = []
        if maxanchor>.0015: reasons.append('Fixed active contact anchor error exceeds 1.5 mm')
        if minfloor<-.001: reasons.append('Ground penetration exceeds 1 mm')
        if selfpen>.001: reasons.append('Self penetration exceeds 1 mm')
        if margin<.005: reasons.append('COM margin below 5 mm in unchanged active support hull')
        if phase==1. and abs(kneegap)>.002: reasons.append('Right knee region did not establish ground contact within 2 mm')
        out={'label':'KINEMATIC/STATIC ONLY; NOT an achieved physical transition', 'phase':float(phase),
             'nfev':0 if sol is None else sol.nfev, 'solver_success':None if sol is None else bool(sol.success),
             'qpos':d.qpos.tolist(), 'joint_pose_rad':dict(zip(info.names,map(float,x[6:]))),
             'base_xyz_m':x[:3].tolist(),'com_xyz_m':com.tolist(),'com_margin_m':margin,
             'max_anchor_error_m':maxanchor,'anchor_errors_xyz_m':error.tolist(),
             'minimum_floor_clearance_m':minfloor,'max_self_penetration_m':selfpen,
             'right_knee_region_clearance_m':kneegap,'right_foot_clearance_m':p.bottom(right_foot)[0],
             'all_clearances_m':allclear,'geometric_pass':not reasons,'failure_reasons':reasons}
        return out

    statics_path = KIN/'statics/check_static_torques.py'
    spec = importlib.util.spec_from_file_location('support_statics',statics_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    points = [{'geom':a['geom'],'body_id':a['body_id'],'position_xyz_m':a['world_goal'],'mu':.8} for a in anchors]
    start = time.monotonic()
    x = x0.copy()
    reports = [report(x,0.,None)]
    for index in range(1,12):
        phase = index/11
        sol = least_squares(residual,x,args=(phase,x.copy()),bounds=(bounds[:,0],bounds[:,1]),
                            max_nfev=100,diff_step=1e-4,ftol=1e-7,xtol=1e-7,gtol=1e-7)
        x = sol.x
        r = report(x,phase,sol)
        q=d.qpos[info.qadr]
        zero=np.zeros(m.nu)
        statics = module.solve(info,d,points,np.column_stack([info.torque(q,zero,info.lower+.05),info.torque(q,zero,info.upper-.05)]))
        r['active_support_static_lp']=statics
        if statics['feasible']:
            torque=np.array([statics['joint_torques_Nm'][n] for n in info.names])
            stiffness=np.maximum(info.kd,info.effort/(.25*info.velocity))*(info.kp/info.kd)
            targets=q+torque/stiffness
            r['static_offset_targets_rad']=dict(zip(info.names,map(float,targets)))
            r['servo_torque_error_Nm']=float(np.max(np.abs(info.torque(q,zero,targets)-torque)))
        reports.append(r)
        (OUT/f'waypoint_{index:02}.json').write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps({k:r[k] for k in ('phase','nfev','geometric_pass','failure_reasons','max_anchor_error_m','minimum_floor_clearance_m','right_knee_region_clearance_m')}),flush=True)
        if not r['geometric_pass']:
            break
    summary={'label':'KINEMATIC/STATIC ONLY; never integrated or injected into runtime',
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'input_state_sha256':hashlib.sha256(sourcepath.read_bytes()).hexdigest(),
             'max_waypoints':12,'max_nfev_each':100,'fixed_schedule':'Both hands and left heel retained; right foot unloaded, right knee region added at last waypoint',
             'anchors':anchors,'initial_physical_qvel_not_used_for_statics':source['qvel'],
             'elapsed_seconds':time.monotonic()-start,'physics_steps':0,
             'num_waypoints':len(reports),'complete_sampled_waypoints':len(reports)==12 and all(r['geometric_pass'] for r in reports),
             'results':reports}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    audit_existing()


def audit_existing():
    """Check existing interpolation; no optimization or physical stepping."""
    path = OUT/'summary.json'
    summary = json.loads(path.read_text())
    # Handle the initial saved run, before the interpolation audit was added.
    summary['complete_sampled_waypoints'] = summary.get('complete_sampled_waypoints', summary.get('complete_geometric_path', False))
    p=Planner()
    m,d,info=p.m,p.d,p.info
    anchors=summary['anchors']
    goals=np.array([a['world_goal'] for a in anchors])
    hull=ConvexHull(goals[:,:2])
    audit=[]
    for interval,(a,b) in enumerate(zip(summary['results'][:-1],summary['results'][1:])):
        qa,qb=np.array(a['qpos']),np.array(b['qpos'])
        rotation=Slerp([0,1],Rotation.from_quat(np.stack([qa[[4,5,6,3]],qb[[4,5,6,3]]])))
        samples=[]
        for f in np.linspace(0,1,9)[1:-1]:
            d.qpos[:]=qa*(1-f)+qb*f
            d.qpos[3:7]=rotation(float(f)).as_quat()[[3,0,1,2]]
            d.qvel[:]=0
            mujoco.mj_forward(m,d)
            actual=np.array([d.xpos[c['body_id']]+d.xmat[c['body_id']].reshape(3,3)@c['local_point'] for c in anchors])
            error=float(np.max(np.linalg.norm(actual-goals,axis=1)))
            clear=min(p.bottom(g)[0] for g in p.ids)
            selfpen=p.self_penetration()
            com=d.subtree_com[m.body('pelvis').id]
            margin=float(-np.max(hull.equations[:,:2]@com[:2]+hull.equations[:,2]))
            samples.append({'fraction':float(f),'anchor_error_m':error,'min_floor_clearance_m':clear,
                            'self_penetration_m':selfpen,'com_margin_m':margin,
                            'pass':bool(error<=.0015 and clear>=-.001 and selfpen<=.001 and margin>=.005)})
        audit.append({'from_waypoint':interval,'to_waypoint':interval+1,
                      'max_joint_step_rad':float(np.max(np.abs(qb[info.qadr]-qa[info.qadr]))),
                      'samples':samples,'pass':all(s['pass'] for s in samples)})
    # The fixed contact schedule adds the right knee at the final waypoint.
    last=summary['results'][-1]
    if last['phase']==1. and abs(last['right_knee_region_clearance_m'])<.002:
        d.qpos[:]=last['qpos'];d.qvel[:]=0;mujoco.mj_forward(m,d)
        spec=importlib.util.spec_from_file_location('support_statics',KIN/'statics/check_static_torques.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        points=[{'geom':c['geom'],'body_id':c['body_id'],'position_xyz_m':c['world_goal'],'mu':.8} for c in anchors]
        kg=p.knee_region[1]
        for point in p.bottom(kg)[1]:
            points.append({'geom':m.geom(kg).name,'body_id':int(m.geom_bodyid[kg]),'position_xyz_m':[float(point[0]),float(point[1]),0.],'mu':.8})
        q=d.qpos[info.qadr];zero=np.zeros(m.nu)
        static=module.solve(info,d,points,np.column_stack([info.torque(q,zero,info.lower+.05),info.torque(q,zero,info.upper-.05)]))
        last['final_knee_added_static_lp']=static
        if static['feasible']:
            torque=np.array([static['joint_torques_Nm'][n] for n in info.names])
            stiffness=np.maximum(info.kd,info.effort/(.25*info.velocity))*(info.kp/info.kd)
            targets=q+torque/stiffness
            last['static_offset_targets_rad']=dict(zip(info.names,map(float,targets)))
            last['servo_torque_error_Nm']=float(np.max(np.abs(info.torque(q,zero,targets)-torque)))
        (OUT/'waypoint_11.json').write_text(json.dumps(last,indent=2)+'\n')
    summary['interpolation_audit']={'samples_per_interval':7,'intervals':audit,'all_intervals_pass':all(a['pass'] for a in audit)}
    summary['complete_geometric_path']=summary['complete_sampled_waypoints'] and all(a['pass'] for a in audit)
    summary['failure_reason']=None if summary['complete_geometric_path'] else 'The single continuation found sampled support poses, but joint-space interpolation between waypoints loses fixed support and/or floor clearance. This is not a usable continuous path.'
    summary['audit_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'continuous_path_pass':summary['complete_geometric_path'],
                      'failed_intervals':[a['from_waypoint'] for a in audit if not a['pass']],
                      'final_knee_added_static_feasible':last.get('final_knee_added_static_lp',{}).get('feasible')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-only',action='store_true')
    args=parser.parse_args()
    if args.audit_only:
        audit_existing()
    else:
        main()
