"""Additional posture diagnostics; the original recovery definition is unchanged."""
import numpy as np
import mujoco

MIN_WIDTH = .16
MAX_WIDTH = .36
MAX_HEADING = np.deg2rad(25.)
MAX_HIP_YAW = .45
MAX_SOLE_TILT = np.deg2rad(20.)
MAX_FOOT_BRACING_FORCE = 2.


def stance_metrics(info, data):
    m = info.model
    pelvis_rotation = data.xmat[m.body('pelvis').id].reshape(3, 3)
    foot_geoms = [m.geom(f'{side}_ankle_roll_link_collision_0').id for side in ('left', 'right')]
    delta = data.geom_xpos[foot_geoms[0]] - data.geom_xpos[foot_geoms[1]]
    pelvis_forward = pelvis_rotation[:2, 0]
    pelvis_forward = pelvis_forward / max(np.linalg.norm(pelvis_forward), 1e-8)
    pelvis_left = np.array([-pelvis_forward[1], pelvis_forward[0]])
    width = float(delta[:2] @ pelvis_left)
    headings, sole_tilts = [], []
    for geom in foot_geoms:
        forward = data.geom_xmat[geom].reshape(3, 3)[:2, 0]
        forward = forward / max(np.linalg.norm(forward), 1e-8)
        headings.append(float(np.arccos(np.clip(forward @ pelvis_forward, -1, 1))))
        sole_tilts.append(float(np.arccos(np.clip(data.geom_xmat[geom].reshape(3, 3)[2, 2], -1, 1))))
    hip_yaw = [float(data.qpos[info.qadr[info.names.index(f'{side}_hip_yaw_joint')]])
               for side in ('left', 'right')]
    bracing_force = 0.
    force = np.empty(6)
    for index in range(data.ncon):
        contact = data.contact[index]
        bodies = {int(m.geom_bodyid[contact.geom1]), int(m.geom_bodyid[contact.geom2])}
        if bodies == set(info.foot_bodies):
            mujoco.mj_contactForce(m, data, index, force)
            bracing_force += max(float(force[0]), 0.)
    checks = {'stance_width': MIN_WIDTH <= width <= MAX_WIDTH,
              'foot_heading': all(angle <= MAX_HEADING for angle in headings),
              'hip_yaw': all(abs(angle) <= MAX_HIP_YAW for angle in hip_yaw),
              'flat_soles': all(angle <= MAX_SOLE_TILT for angle in sole_tilts),
              'no_foot_bracing': bracing_force <= MAX_FOOT_BRACING_FORCE}
    return {'signed_foot_width_m': width, 'foot_heading_error_rad': headings,
            'hip_yaw_rad': hip_yaw, 'sole_tilt_rad': sole_tilts, 'foot_bracing_force_n': bracing_force,
            'checks': checks, 'posture_ok': all(checks.values())}
