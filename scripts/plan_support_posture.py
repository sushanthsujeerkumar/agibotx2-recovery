#!/usr/bin/env python3
"""Bounded OFFLINE KINEMATIC ONLY support-posture proposals.

No mj_step, recovery rollout, policy training, or physical state injection. These
static geometric proposals require separate torque/contact feasibility testing.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import time

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

from x2_recovery.common import ModelInfo


class Planner:
    def __init__(self):
        self.info = ModelInfo(physics_profile="guarded_v2")
        self.m = self.info.model
        self.d = mujoco.MjData(self.m)
        self.ids = np.flatnonzero((self.m.geom_contype != 0) & (self.m.geom_bodyid != 0))
        self.feet = [self.m.geom(f"{s}_ankle_roll_link_collision_0").id for s in ("left", "right")]
        self.knees = [self.m.geom(f"{s}_knee_link_collision_0").id for s in ("left", "right")]
        self.knee_region = [self.m.geom(f"{s}_hip_yaw_link_collision_0").id for s in ("left", "right")]
        self.hands = [self.m.geom(f"{s}_wrist_roll_link_collision_0").id for s in ("left", "right")]
        # z, base pitch, left/right hip pitch, knee, ankle, waist pitch,
        # shoulder pitch/roll, elbow, wrist pitch/roll. Arms are mirrored.
        self.names = ["z", "base_pitch", "left_hip_pitch_joint", "left_knee_joint",
                      "left_ankle_pitch_joint", "right_hip_pitch_joint", "right_knee_joint",
                      "right_ankle_pitch_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
                      "left_shoulder_roll_joint", "left_elbow_joint", "left_wrist_pitch_joint",
                      "left_wrist_roll_joint"]
        limits = {n: (self.info.lower[i]+.05, self.info.upper[i]-.05) for i, n in enumerate(self.info.names)}
        self.bounds = np.array([(.04, .65), (-.7, 1.7)] + [limits[n] for n in self.names[2:]])

    def contacts(self, kind):
        if kind == "hands_feet":
            return self.feet + self.hands
        if kind == "hands_knees":
            return self.knees + self.hands
        if kind == "hands_knee_region":
            return self.knee_region + self.hands
        if kind == "half_kneel":
            return [self.feet[0], self.knees[1]] + self.hands
        if kind == "half_knee_region":
            return [self.feet[0], self.knee_region[1]] + self.hands
        if kind == "seat_feet_hands":
            return self.feet + self.hands + [self.m.geom("pelvis_collision_0").id]
        raise ValueError(kind)

    def pose(self, x):
        self.d.qpos[:] = self.info.standing
        self.d.qpos[:3] = [0., 0., x[0]]
        self.d.qpos[3:7] = [np.cos(x[1]/2), 0., np.sin(x[1]/2), 0.]
        for n, v in zip(self.names[2:], x[2:]):
            self.d.qpos[self.m.joint(n).qposadr] = v
        for suffix, sign in [("shoulder_pitch_joint", 1), ("shoulder_roll_joint", -1),
                             ("elbow_joint", 1), ("wrist_pitch_joint", 1), ("wrist_roll_joint", -1)]:
            self.d.qpos[self.m.joint("right_"+suffix).qposadr] = sign * self.d.qpos[self.m.joint("left_"+suffix).qposadr]
        mujoco.mj_forward(self.m, self.d)

    def bottom(self, g):
        pos = self.d.geom_xpos[g]
        rot = self.d.geom_xmat[g].reshape(3, 3)
        size = self.m.geom_size[g]
        typ = self.m.geom_type[g]
        if typ == mujoco.mjtGeom.mjGEOM_BOX:
            points = np.array(list(itertools.product(*[(-s, s) for s in size]))) @ rot.T + pos
            return float(points[:, 2].min()), points[points[:, 2] < points[:, 2].min() + .001]
        if typ == mujoco.mjtGeom.mjGEOM_CAPSULE:
            points = np.stack((pos - rot[:, 2]*size[1], pos + rot[:, 2]*size[1]))
            points[:, 2] -= size[0]
            return float(points[:, 2].min()), points[points[:, 2] < points[:, 2].min()+.001]
        if typ == mujoco.mjtGeom.mjGEOM_SPHERE:
            point = pos - [0, 0, size[0]]
            return float(point[2]), np.array([point])
        raise ValueError(f"Unhandled collision type {typ}")

    def self_penetration(self):
        return max([0.] + [-float(c.dist) for c in self.d.contact
                           if c.geom1 not in self.info.floor_geoms and c.geom2 not in self.info.floor_geoms])

    def residual(self, x, kind, ref):
        self.pose(x)
        contacts = self.contacts(kind)
        bottoms = {g: self.bottom(g) for g in self.ids}
        support = np.concatenate([bottoms[g][1] for g in contacts])
        com = self.d.subtree_com[self.m.body("pelvis").id]
        low = support[:, :2].min(axis=0) + .025
        high = support[:, :2].max(axis=0) - .025
        result = [80*bottoms[g][0] for g in contacts]
        result.extend(150 * min(0., bottoms[g][0]) for g in self.ids)
        result.extend(30*np.maximum(low-com[:2], 0.))
        result.extend(30*np.maximum(com[:2]-high, 0.))
        result.append(150*self.self_penetration())
        # Penalize hands behind knees/feet, leaving space to shift COM forward.
        if kind != "seat_feet_hands":
            result.append(3*max(0., .08 + np.mean([bottoms[g][1][0, 0] for g in contacts[:2]]) - np.mean([bottoms[g][1][0, 0] for g in self.hands])))
        if not kind.startswith("half_"):
            result.extend(2*(x[2:5]-x[5:8]))
        if kind == "hands_feet":
            for g in self.feet:
                result.extend(4*self.d.geom_xmat[g].reshape(3, 3)[2, :2])
        result.extend(.012*(x-ref))
        return np.asarray(result)

    def report(self, x, kind, result):
        self.pose(x)
        contacts = self.contacts(kind)
        bottoms = {g: self.bottom(g) for g in self.ids}
        support = np.concatenate([bottoms[g][1] for g in contacts])
        com = self.d.subtree_com[self.m.body("pelvis").id]
        hull = ConvexHull(support[:, :2])
        margin = float(-np.max(hull.equations[:, :2] @ com[:2] + hull.equations[:, 2]))
        target = self.d.qpos[self.info.qadr]
        joint_margin = float(np.min(np.minimum(target-self.info.lower-.05, self.info.upper-.05-target)))
        ground_min = min(v[0] for v in bottoms.values())
        contact_error = max(abs(bottoms[g][0]) for g in contacts)
        self_pen = self.self_penetration()
        return {
            "label": "KINEMATIC ONLY; NOT physical feasibility or recovery success",
            "candidate": kind, "solver_success": bool(result.success), "nfev": result.nfev,
            "variable_bounds": dict(zip(self.names, self.bounds.tolist())),
            "cost": float(result.cost), "variables": dict(zip(self.names, map(float, x))),
            "qpos": self.d.qpos.tolist(), "joint_targets_rad": dict(zip(self.info.names, map(float, target))),
            "base_xyz_m": self.d.qpos[:3].tolist(), "base_quat_wxyz": self.d.qpos[3:7].tolist(),
            "com_xyz_m": com.tolist(), "support_polygon_xy_m": support[hull.vertices, :2].tolist(),
            "com_support_margin_m": margin, "ground_min_clearance_m": ground_min,
            "max_proposed_contact_error_m": contact_error, "max_self_penetration_m": self_pen,
            "min_guarded_joint_margin_rad": joint_margin,
            "geometric_candidate_pass": bool(ground_min >= -.002 and contact_error < .003 and self_pen < .002 and margin > .005 and joint_margin >= -1e-8),
            "proposed_contacts": [{"geom": self.m.geom(g).name, "lowest_z_m": bottoms[g][0],
                                   "lowest_points_xyz_m": bottoms[g][1].tolist()} for g in contacts],
            "all_geom_clearances_m": {self.m.geom(g).name: v[0] for g, v in bottoms.items()},
            "limitations": ["Only a static qpos geometry calculation, with zero velocities.",
                            "No contact-force, friction, torque-limit or reachability guarantee.",
                            "Support uses collision primitive lowest points; visual mesh is not a contact surface.",
                            "A physical transition must establish contacts before COM leaves existing support."],
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation/floor_transition/kinematics"))
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--stage", choices=("initial", "refined"), default="initial")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    p = Planner()
    seeds = {
        "hands_feet": [.32, .8, -1.6, 2.1, -.7, -1.6, 2.1, -.7, .15, -.35, .15, -.2, 0., 0.],
        "hands_knees": [.34, .8, -.9, 1.7, 0., -.9, 1.7, 0., .15, -.35, .15, -.2, 0., 0.],
        "half_kneel": [.35, .8, -1.7, 1.8, -.7, -.9, 1.7, 0., .15, -.35, .15, -.2, 0., 0.],
    }
    if args.stage == "refined":
        seeds = {
            "hands_knee_region": [.35, .9786, -.6146, 1.4393, .02, -.6146, 1.4393, .02, .0486, -.7934, .1292, -.4842, -.0517, -.1488],
            "half_knee_region": [.3616, .9821, -2.3673, 2.0687, -.6160, -.7150, 1.5344, .025, .0590, -.8064, .0567, -.4559, .0247, -.3306],
            "seat_feet_hands": [.085, -.2, -2.4, 2.3, .35, -2.4, 2.3, .35, -.1, 1.1, .15, -1.7, 0., 0.],
        }
    reports = []
    start = time.monotonic()
    for kind, seed in seeds.items():
        ref = np.clip(seed, p.bounds[:, 0]+1e-6, p.bounds[:, 1]-1e-6)
        result = least_squares(p.residual, ref, args=(kind, ref), bounds=(p.bounds[:, 0], p.bounds[:, 1]), max_nfev=350, diff_step=1e-4)
        report = p.report(result.x, kind, result)
        (args.output/f"{kind}.json").write_text(json.dumps(report, indent=2)+"\n")
        reports.append(report)
        print(json.dumps({k: report[k] for k in ("candidate", "nfev", "geometric_candidate_pass", "base_xyz_m", "com_support_margin_m", "ground_min_clearance_m", "max_proposed_contact_error_m", "max_self_penetration_m")}), flush=True)
        if args.render:
            import imageio.v3 as iio
            with mujoco.Renderer(p.m, height=720, width=960) as renderer:
                camera = mujoco.MjvCamera()
                camera.lookat[:] = [0., 0., .35]
                camera.distance = 1.8
                camera.azimuth = 120
                camera.elevation = -12
                renderer.update_scene(p.d, camera=camera)
                iio.imwrite(args.output/f"{kind}_KINEMATIC_ONLY.png", renderer.render())
    (args.output/f"summary_{args.stage}.json").write_text(json.dumps({
        "label": "KINEMATIC ONLY", "elapsed_seconds": time.monotonic()-start,
        "physics_stepping_performed": False, "stage": args.stage,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "max_nfev_each": 350, "num_candidates": len(reports), "results": reports,
    }, indent=2)+"\n")


if __name__ == "__main__":
    main()
