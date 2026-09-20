"""Shared model, observations, actuator control and success specification."""
from pathlib import Path
import json
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "assets/x2/scene.xml"
CONTROL_DT = 0.02
EPISODE_SECONDS = 15.0
HOLD_SECONDS = 2.0
SUCCESS_TILT = np.deg2rad(15.0)
SUCCESS_LINEAR_SPEED = 0.15
SUCCESS_ANGULAR_SPEED = 0.3
CONTACT_FORCE = 2.0


class ModelInfo:
    def __init__(self, model=None, physics_profile="legacy"):
        self.model = model or mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        m = self.model
        meta_path = MODEL_PATH.parent / "model_metadata.json"
        self.metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        self.joint_ids = m.actuator_trnid[:, 0].astype(int)
        from .physics import configure_model
        self.physics_profile = physics_profile
        configure_model(m, self.joint_ids, physics_profile)
        self.names = [m.joint(int(i)).name for i in self.joint_ids]
        self.qadr = m.jnt_qposadr[self.joint_ids]
        self.vadr = m.jnt_dofadr[self.joint_ids]
        self.lower = m.jnt_range[self.joint_ids, 0].copy()
        self.upper = m.jnt_range[self.joint_ids, 1].copy()
        self.effort = np.max(np.abs(m.actuator_ctrlrange), axis=1)
        self.velocity = np.asarray(self.metadata.get("velocity_limits", [12.] * m.nu))
        self.kp = np.asarray(self.metadata.get("kp", [70. if any(k in n for k in ("hip", "knee", "waist")) else 30. for n in self.names]))
        self.kd = np.asarray(self.metadata.get("kd", [3. if any(k in n for k in ("hip", "knee", "waist")) else 1.5 for n in self.names]))
        self.standing = self._key("standing")
        self.supine = self._key("supine")
        self.nominal = self.standing[self.qadr].copy()
        self.action_scale = np.minimum((self.upper - self.lower) * 0.5, 2.0)
        self.action_positive = self.upper - self.nominal
        self.action_negative = self.nominal - self.lower
        self.height_threshold = 0.9 * self.standing[2]
        self.torso_id = m.body("torso_link").id
        self.foot_bodies = [m.body(f"{s}_ankle_roll_link").id for s in ("left", "right")]
        self.floor_geoms = set(np.flatnonzero((m.geom_bodyid == 0) & (m.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)).tolist())
        self.substeps = int(round(CONTROL_DT / m.opt.timestep))
        if not np.isclose(self.substeps * m.opt.timestep, CONTROL_DT):
            raise ValueError("Control period must be an integer multiple of physics timestep")

    def _key(self, name):
        try:
            return self.model.key(name).qpos.copy()
        except KeyError as e:
            raise RuntimeError(f"Prepared model requires {name} keyframe") from e

    def targets(self, action):
        action = np.clip(action, -1, 1)
        scale = np.where(action >= 0, self.action_positive, self.action_negative)
        return np.clip(self.nominal + action * scale, self.lower, self.upper)

    def torque(self, q, v, target):
        if self.physics_profile == "guarded_v2":
            from .physics import guarded_torque
            return guarded_torque(self, q, v, target)
        torque = np.clip(self.kp * (target - q) - self.kd * v, -self.effort, self.effort)
        # Never command further acceleration beyond the published speed limit.
        torque = np.where((v >= self.velocity) & (torque > 0), 0., torque)
        return np.where((v <= -self.velocity) & (torque < 0), 0., torque)


def ground_forces(info, data):
    """Actual solver normal forces; ignore self contact when checking ground support."""
    values = np.zeros(3)
    force = np.empty(6)
    for c in range(data.ncon):
        contact = data.contact[c]
        g1, g2 = int(contact.geom1), int(contact.geom2)
        if g1 in info.floor_geoms:
            body = int(info.model.geom_bodyid[g2])
        elif g2 in info.floor_geoms:
            body = int(info.model.geom_bodyid[g1])
        else:
            continue
        mujoco.mj_contactForce(info.model, data, c, force)
        idx = info.foot_bodies.index(body) if body in info.foot_bodies else 2
        values[idx] += max(float(force[0]), 0.)
    return values


def observation_numpy(info, data, previous_action, forces):
    rotation = data.xmat[info.model.body("pelvis").id].reshape(3, 3)
    obs = np.concatenate([
        data.qpos[info.qadr] - info.nominal,
        data.qvel[info.vadr] * 0.1,
        rotation.T @ np.array([0., 0., -1.]),
        rotation.T @ data.qvel[:3],
        data.qvel[3:6] * 0.25,
        [data.qpos[2]],
        (forces > CONTACT_FORCE).astype(float),
        previous_action,
    ])
    return np.clip(obs, -10., 10.).astype(np.float32)


def sensor_forces(info, data):
    """Same touch features used by GPU training (may include self contact)."""
    m = info.model
    feet = [float(data.sensordata[m.sensor_adr[i]]) for i in info.metadata["foot_sensor_indices"]]
    other = sum(float(data.sensordata[m.sensor_adr[i]]) for i in info.metadata["nonfoot_sensor_indices"])
    return np.asarray(feet + [other])


def success_conditions(info, data, forces):
    torso_up = float(data.xmat[info.torso_id].reshape(3, 3)[2, 2])
    return {
        "upright": torso_up >= float(np.cos(SUCCESS_TILT)),
        "height": bool(data.qpos[2] >= info.height_threshold),
        "both_feet": bool(np.all(forces[:2] > CONTACT_FORCE)),
        "no_other_support": bool(forces[2] <= CONTACT_FORCE),
        "low_linear_speed": bool(np.linalg.norm(data.qvel[:3]) < SUCCESS_LINEAR_SPEED),
        "low_angular_speed": bool(np.linalg.norm(data.qvel[3:6]) < SUCCESS_ANGULAR_SPEED),
    }


def reset_cpu(info, data, seed=0):
    """Settle only a supine initial state; no standing-start curriculum."""
    rng = np.random.default_rng(seed)
    mujoco.mj_resetData(info.model, data)
    data.qpos[:] = info.supine
    # Small in-plane translation does not alter the floor clearance or posture.
    data.qpos[:2] += rng.uniform(-0.02, 0.02, 2)
    data.qpos[info.qadr] = np.clip(data.qpos[info.qadr] + rng.uniform(-.005, .005, len(info.qadr)), info.lower, info.upper)
    data.qpos[2] += .005
    data.qvel[:] = 0
    mujoco.mj_forward(info.model, data)
    target = data.qpos[info.qadr].copy()
    for _ in range(round(1.0 / info.model.opt.timestep)):
        data.ctrl[:] = info.torque(data.qpos[info.qadr], data.qvel[info.vadr], target)
        mujoco.mj_step(info.model, data)
    # Settling is outside the timed episode; retain physically attained velocities.
    data.time = 0.
    mujoco.mj_forward(info.model, data)
    if not np.isfinite(data.qpos).all() or np.max(np.abs(data.qvel)) > 20:
        raise RuntimeError("Supine settling is unstable")
