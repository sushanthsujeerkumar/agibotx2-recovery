#!/usr/bin/env python3
"""Reproducibly prepare the official X2 Ultra 1.3 model for recovery.

Only the Python standard library and git are required. Physical inertias,
kinematic transforms, limits and visual meshes are taken from the pinned URDF.
Collision primitives and simulation/controller parameters are project choices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "https://github.com/AgibotTech/agibot_x2_urdf.git"
REVISION = "60c5de582c523cd188f563819e62d34cfdc3d2d0"
VERSION = "X2_URDF-v1.3.0"
URDF_NAME = "x2_ultra_simple_collision.urdf"


def values(text):
    return [float(x) for x in text.split()]


def fmt(numbers):
    return " ".join(f"{x:.12g}" for x in numbers)


def quat_rpy(rpy):
    r, p, y = [v / 2 for v in values(rpy)]
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return fmt([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def origin_attrs(element):
    origin = element.find("origin")
    if origin is None:
        return {}
    return {"pos": origin.get("xyz", "0 0 0"), "quat": quat_rpy(origin.get("rpy", "0 0 0"))}


def collision_shapes(link):
    """Primitive approximations, measured against vendor mesh dimensions (m)."""
    if link == "pelvis":
        return [dict(type="box", pos="-0.005 0 0.002", size="0.070 0.083 0.063")]
    if link == "torso_link":
        return [dict(type="box", pos="0.010 0 0.182", size="0.092 0.124 0.133"),
                dict(type="box", pos="-0.126 0 0.196", size="0.049 0.105 0.108")]
    if link == "waist_yaw_link":
        return [dict(type="capsule", fromto="0 0 0.018 0 0 0.078", size="0.055")]
    if link == "head_pitch_link":
        return [dict(type="sphere", pos="-0.002 0 0.001", size="0.081")]
    side = 1 if link.startswith("left") else -1
    if "hip_pitch_link" in link:
        return [dict(type="capsule", fromto=f"0 {side * .02} 0 0 {side * .086} 0", size="0.036")]
    if "hip_roll_link" in link:
        return [dict(type="capsule", fromto="0 0 -0.008 0 0 -0.103", size="0.049")]
    if "hip_yaw_link" in link:
        return [dict(type="capsule", fromto="0.004 0 -0.044 -0.009 0 -0.193", size="0.048")]
    if "knee_link" in link:
        return [dict(type="capsule", fromto="0.002 0 -0.003 -0.007 0 -0.251", size="0.046")]
    if "ankle_pitch_link" in link:
        return [dict(type="sphere", size="0.032")]
    if "ankle_roll_link" in link:
        return [dict(type="box", pos="0.036 0 -0.0494", size="0.106 0.060 0.024")]
    if "shoulder_pitch_link" in link:
        return [dict(type="capsule", fromto=f"0 {side * .014} 0 0 {side * .068} 0", size="0.030")]
    if "shoulder_roll_link" in link:
        return [dict(type="capsule", fromto="0 0 -0.007 0 0 -0.090", size="0.039")]
    if "shoulder_yaw_link" in link:
        return [dict(type="capsule", fromto="0.003 0 -0.019 0.009 0 -0.075", size="0.035")]
    if "elbow_link" in link:
        return [dict(type="capsule", fromto="-0.012 0 -0.004 -0.012 0 -0.090", size="0.034")]
    if "wrist_yaw_link" in link:
        return [dict(type="capsule", fromto="0 0 -0.024 0 0 -0.063", size="0.033")]
    if "wrist_roll_link" in link:
        return [dict(type="box", pos="0 0 -0.1", size="0.035 0.0225 0.07")]
    return []


def nominal_joint(name):
    if "hip_pitch" in name:
        return -0.12
    if "knee" in name:
        return 0.24
    if "ankle_pitch" in name:
        return -0.12
    if "shoulder_roll" in name:
        return 0.08 if name.startswith("left") else -0.08
    if "elbow" in name:
        return -0.12
    return 0.0


def gains(name):
    if any(term in name for term in ("hip_", "knee", "waist_yaw")):
        return 90.0, 4.0
    if "ankle" in name:
        return 40.0, 2.0
    if "waist" in name:
        return 60.0, 3.0
    if "wrist_pitch" in name or "wrist_roll" in name:
        return 8.0, 0.5
    if "head" in name:
        return 3.0, 0.3
    return 30.0, 1.5


def prepare(source: Path, destination: Path):
    resolved_revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if resolved_revision != REVISION:
        raise ValueError(f"Expected source revision {REVISION}, got {resolved_revision}")
    folder = source / VERSION
    urdf = ET.parse(folder / URDF_NAME).getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {}
    for joint in urdf.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "meshes").mkdir(exist_ok=True)
    mj = ET.Element("mujoco", model="x2_ultra_recovery")
    ET.SubElement(mj, "compiler", angle="radian", meshdir="meshes", autolimits="true", inertiafromgeom="false")
    ET.SubElement(mj, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast", solver="Newton", iterations="10", ls_iterations="10", cone="pyramidal")
    ET.SubElement(mj, "statistic", center="0 0 0.6", extent="1.7")
    visual = ET.SubElement(mj, "visual")
    ET.SubElement(visual, "headlight", diffuse="0.7 0.7 0.7", ambient="0.3 0.3 0.3")
    ET.SubElement(visual, "global", azimuth="135", elevation="-20", offwidth="1280", offheight="720")
    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", damping="0.1", armature="0.02", frictionloss="0.1", limited="true")
    collision_default = ET.SubElement(default, "default", {"class": "collision"})
    ET.SubElement(collision_default, "geom", group="3", contype="1", conaffinity="1", condim="3", friction="0.8 0.005 0.0001", solref="0.008 1", solimp="0.9 0.95 0.001", density="0", rgba="0.4 0.65 0.85 0.25")
    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "texture", name="floor_texture", type="2d", builtin="checker", rgb1="0.14 0.18 0.22", rgb2="0.20 0.25 0.30", width="512", height="512")
    ET.SubElement(asset, "material", name="floor_material", texture="floor_texture", texrepeat="4 4", reflectance="0.05")
    world = ET.SubElement(mj, "worldbody")
    ET.SubElement(world, "light", pos="0 -1 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="floor", type="plane", size="0 0 0.1", material="floor_material", friction="0.8 0.005 0.0001", condim="3")
    actuators = ET.Element("actuator")
    contact = ET.Element("contact")
    sensors = ET.Element("sensor")
    joints_meta = []
    touch_sensor_names = []
    touch_sensor_bodies = []
    mesh_names = {}
    parents = {}
    colliding_bodies = set()

    def add_link(name, parent_element, joint=None):
        attrs = {"name": name}
        if joint is None:
            attrs["pos"] = "0 0 0.672"
        else:
            attrs.update(origin_attrs(joint))
        body = ET.SubElement(parent_element, "body", attrs)
        link = links[name]
        if joint is None:
            ET.SubElement(body, "freejoint", name="floating_base_joint")
        elif joint.get("type") != "fixed":
            limit = joint.find("limit")
            jname = joint.get("name")
            lower, upper, effort, velocity = [float(limit.get(k)) for k in ("lower", "upper", "effort", "velocity")]
            ET.SubElement(body, "joint", name=jname, type="hinge", axis=joint.find("axis").get("xyz"), range=fmt([lower, upper]), actuatorfrcrange=fmt([-effort, effort]))
            ET.SubElement(actuators, "motor", name=f"motor_{jname}", joint=jname, gear="1", ctrllimited="true", ctrlrange=fmt([-effort, effort]), forcelimited="true", forcerange=fmt([-effort, effort]))
            kp, kd = gains(jname)
            joints_meta.append(dict(name=jname, body=name, lower=lower, upper=upper, effort=effort, velocity=velocity, kp=kp, kd=kd, nominal=nominal_joint(jname)))
        inertia = link.find("inertial")
        if inertia is not None:
            inertial_attrs = origin_attrs(inertia)
            # The supplied tensors are in link axes; all inertia origins have zero RPY.
            if any(abs(v) > 1e-10 for v in values(inertia.find("origin").get("rpy", "0 0 0"))):
                raise ValueError(f"Rotated inertia needs explicit tensor rotation: {name}")
            inertial_attrs.pop("quat", None)
            inertial_attrs["mass"] = inertia.find("mass").get("value")
            tensor = inertia.find("inertia")
            inertial_attrs["fullinertia"] = " ".join(tensor.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz"))
            ET.SubElement(body, "inertial", inertial_attrs)
        for index, item in enumerate(link.findall("visual")):
            mesh = item.find("geometry/mesh")
            if mesh is None:
                continue
            filename = Path(mesh.get("filename")).name
            scale = mesh.get("scale", "1 1 1")
            key = (filename, scale)
            if key not in mesh_names:
                mesh_names[key] = f"mesh_{len(mesh_names)}"
                ET.SubElement(asset, "mesh", name=mesh_names[key], file=filename, scale=scale)
                shutil.copy2(folder / "meshes" / filename, destination / "meshes" / filename)
            attrs = dict(name=f"{name}_visual_{index}", type="mesh", mesh=mesh_names[key], group="2", contype="0", conaffinity="0", density="0")
            attrs.update(origin_attrs(item))
            color = item.find("material/color")
            attrs["rgba"] = color.get("rgba") if color is not None else "0.8 0.82 0.84 1"
            ET.SubElement(body, "geom", attrs)
        shapes = collision_shapes(name)
        for index, shape in enumerate(shapes):
            ET.SubElement(body, "geom", {"name": f"{name}_collision_{index}", "class": "collision", **shape})
            colliding_bodies.add(name)
        if shapes:
            bounds_min, bounds_max = [float("inf")] * 3, [float("-inf")] * 3
            for shape in shapes:
                size = values(shape["size"])
                if shape["type"] == "capsule":
                    ends = values(shape["fromto"])
                    shape_min = [min(ends[a], ends[a + 3]) - size[0] for a in range(3)]
                    shape_max = [max(ends[a], ends[a + 3]) + size[0] for a in range(3)]
                else:
                    position = values(shape.get("pos", "0 0 0"))
                    halfsize = size if len(size) == 3 else size * 3
                    shape_min = [position[a] - halfsize[a] for a in range(3)]
                    shape_max = [position[a] + halfsize[a] for a in range(3)]
                bounds_min = [min(bounds_min[a], shape_min[a]) for a in range(3)]
                bounds_max = [max(bounds_max[a], shape_max[a]) for a in range(3)]
            sensor_name = f"touch_{name}"
            ET.SubElement(body, "site", name=f"touch_site_{name}", type="box", pos=fmt([(bounds_min[a] + bounds_max[a]) / 2 for a in range(3)]), size=fmt([(bounds_max[a] - bounds_min[a]) / 2 + .005 for a in range(3)]), group="5", rgba="0.3 1 0.3 0.1")
            ET.SubElement(sensors, "touch", name=sensor_name, site=f"touch_site_{name}")
            touch_sensor_names.append(sensor_name)
            touch_sensor_bodies.append(name)
        for child_joint in children.get(name, []):
            child_name = child_joint.find("child").get("link")
            parents[child_name] = name
            add_link(child_name, body, child_joint)

    add_link("pelvis", world)
    # Adjacent joint assemblies overlap by design. Exclude only two-edge
    # ancestors; distant limbs and trunk retain self-collision.
    exclusions = []
    for child in sorted(colliding_bodies):
        ancestor = child
        for _ in range(2):
            ancestor = parents.get(ancestor)
            if ancestor is None:
                break
            if ancestor in colliding_bodies:
                ET.SubElement(contact, "exclude", body1=child, body2=ancestor)
                exclusions.append([child, ancestor])
    mj.append(contact)
    mj.append(actuators)
    mj.append(sensors)
    nominal = [j["nominal"] for j in joints_meta]
    # A negative 90-degree pitch makes the robot's forward (+X) face upward.
    supine = [0, 0, 0.195, math.sqrt(0.5), 0, -math.sqrt(0.5), 0] + nominal
    standing = [0, 0, 0.672, 1, 0, 0, 0] + nominal
    keyframe = ET.SubElement(mj, "keyframe")
    ET.SubElement(keyframe, "key", name="standing", qpos=fmt(standing))
    ET.SubElement(keyframe, "key", name="supine", qpos=fmt(supine))
    ET.indent(mj, space="  ")
    ET.ElementTree(mj).write(destination / "scene.xml", encoding="utf-8", xml_declaration=True)
    shutil.copy2(source / "LICENSE", destination / "LICENSE-MulanPSL-2.0.txt")
    shutil.copy2(folder / URDF_NAME, destination / URDF_NAME)
    metadata = dict(source_repository=REPOSITORY, source_revision=REVISION, source_version=VERSION,
                    source_urdf=URDF_NAME, source_urdf_sha256=hashlib.sha256((folder / URDF_NAME).read_bytes()).hexdigest(),
                    model_sha256=hashlib.sha256((destination / "scene.xml").read_bytes()).hexdigest(),
                    model="scene.xml", mass_kg=sum(float(l.find("inertial/mass").get("value")) for l in links.values() if l.find("inertial/mass") is not None),
                    joint_order=[j["name"] for j in joints_meta], joint_names=[j["name"] for j in joints_meta], joints=joints_meta,
                    velocity_limits=[j["velocity"] for j in joints_meta], kp=[j["kp"] for j in joints_meta], kd=[j["kd"] for j in joints_meta],
                    qpos_order="free base xyz, quaternion wxyz, joint_order", actuator_order="joint_order",
                    nominal_qpos=standing, standing_qpos=standing, supine_qpos=supine, timestep=0.002,
                    feet_bodies=["left_ankle_roll_link", "right_ankle_roll_link"],
                    torso_body="torso_link", base_body="pelvis", collision_exclusions=exclusions,
                    touch_sensor_names=touch_sensor_names, touch_sensor_bodies=touch_sensor_bodies,
                    touch_sensor_indices={name: i for i, name in enumerate(touch_sensor_names)},
                    foot_sensor_names=["touch_left_ankle_roll_link", "touch_right_ankle_roll_link"],
                    foot_sensor_indices=[touch_sensor_names.index(f"touch_{side}_ankle_roll_link") for side in ("left", "right")],
                    nonfoot_sensor_indices=[i for i, name in enumerate(touch_sensor_names) if name not in ("touch_left_ankle_roll_link", "touch_right_ankle_roll_link")],
                    modifications=["URDF is authoritative for all link inertias, kinematic transforms, effort/velocity/position limits.",
                                   "Primitive collision approximations replace mesh collisions; official visual meshes retained.",
                                   "Added flat floor, free base, torque actuators and candidate standing/supine keyframes.",
                                   "PD gains, damping, armature, friction and contact settings are simulation assumptions, not vendor calibration.",
                                   "Velocity values are controller limits; MuJoCo joints have no hard velocity limit. Never clamp simulated qvel.",
                                   "Touch sensors enclose all collision primitives per body and include self-contact; zero nonfoot touch is a conservative success criterion.",
                                   "Supine keyframe starts slightly above floor; settle with controlled dynamics before each episode."])
    (destination / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {destination / 'scene.xml'}: {len(joints_meta)} actuators, {metadata['mass_kg']:.6f} kg")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing vendor git checkout at the pinned revision")
    parser.add_argument("--output", type=Path, default=ROOT / "assets/x2")
    parser.add_argument("--validate", action="store_true", help="Run a 5-second supine physics check (requires mujoco and numpy)")
    args = parser.parse_args()
    source = args.source or ROOT / ".cache/agibot_x2_urdf"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", REPOSITORY, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", REVISION], check=True)
    prepare(source, args.output)
    if args.validate:
        validate(args.output)


def validate(destination):
    """Physics smoke check, deliberately separate from recovery success."""
    import mujoco
    import numpy as np

    meta = json.loads((destination / "model_metadata.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(destination / "scene.xml"))
    data = mujoco.MjData(model)
    joint_ids = model.actuator_trnid[:, 0]
    qpos_ids = model.jnt_qposadr[joint_ids]
    dof_ids = model.jnt_dofadr[joint_ids]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in joint_ids]
    assert names == meta["joint_names"], "Actuator and metadata joint orders differ"
    assert abs(float(model.body_mass.sum()) - meta["mass_kg"]) < 1e-8
    assert model.nq == 38 and model.nv == 37 and model.nu == 31
    mujoco.mj_resetDataKeyframe(model, data, model.key("supine").id)
    mujoco.mj_forward(model, data)
    initial_contacts = int(data.ncon)
    target = np.asarray(meta["supine_qpos"])[qpos_ids]
    kp, kd = np.asarray(meta["kp"]), np.asarray(meta["kd"])
    limit = model.actuator_ctrlrange[:, 1]
    max_depth, max_speed = 0.0, 0.0
    for _ in range(round(5 / model.opt.timestep)):
        data.ctrl[:] = np.clip(kp * (target - data.qpos[qpos_ids]) - kd * data.qvel[dof_ids], -limit, limit)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        assert np.all(np.abs(data.actuator_force) <= limit + 1e-6)
        max_speed = max(max_speed, float(np.max(np.abs(data.qvel[dof_ids]))))
        if data.ncon:
            max_depth = max(max_depth, -min(c.dist for c in data.contact))
    final_depth = max(0.0, -min((c.dist for c in data.contact), default=0.0))
    assert final_depth < 0.002, f"Persistent penetration exceeds 2 mm: {final_depth}"
    assert float(np.linalg.norm(data.qvel)) < .1, "Supine pose did not settle"
    report = dict(mujoco_version=mujoco.__version__, model_sha256=meta["model_sha256"],
                  mass_kg=float(model.body_mass.sum()), nq=model.nq, nv=model.nv, nu=model.nu,
                  initial_contact_count=initial_contacts, duration_simulated_seconds=float(data.time),
                  finite=True, force_limits_respected=True, maximum_transient_contact_depth_m=max_depth,
                  final_contact_depth_m=final_depth, maximum_joint_speed_rad_s=max_speed,
                  final_generalized_velocity_norm=float(np.linalg.norm(data.qvel)),
                  settled_supine_qpos=data.qpos.tolist(), settled_supine_qvel=data.qvel.tolist(),
                  notes="This validates stable supine physics and actuator mapping, not successful recovery or open-loop standing stability.")
    (destination / "model_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("settled_")}, indent=2))


if __name__ == "__main__":
    main()
