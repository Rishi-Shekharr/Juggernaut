"""Generate a static V7 pitch-workspace report; no simulator or policy execution."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parent


def vector(text, default=(0.0, 0.0, 0.0)):
    return np.fromstring(text, sep=" ") if text else np.asarray(default, dtype=float)


def transform(element):
    result = np.eye(4)
    result[:3, 3] = vector(element.get("pos"))
    if element.get("quat"):
        wxyz = vector(element.get("quat"))
        result[:3, :3] = Rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
    elif element.get("euler"):
        # MJCF uppercase XYZ denotes fixed axes; SciPy lowercase xyz matches it.
        result[:3, :3] = Rotation.from_euler("xyz", vector(element.get("euler"))).as_matrix()
    return result


def build_report():
    xml = ROOT / "JUGGERNAUT-v7.xml"
    tree = ET.parse(xml).getroot()
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    bodies = {body.get("name"): body for body in tree.iter("body")}
    parents = {child.get("name"): body for body in bodies.values() for child in body.findall("body")}
    rows = []
    for side, foot_name in zip(("left", "right"), cfg["env"]["foot_links"]):
        foot = bodies[foot_name]
        chain, body = [], foot
        while body is not None:
            chain.append(body)
            body = parents.get(body.get("name"))
        chain.reverse()
        sole = next(geom for geom in foot.findall("geom") if geom.get("name") == side + "_sole_collision")
        signs = np.asarray([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        vertices = signs * vector(sole.get("size"))
        geom_transform = transform(sole)
        vertices = vertices @ geom_transform[:3, :3].T + geom_transform[:3, 3]
        for offset in (0.0, 0.3, 0.6, 0.9):
            angles = {side + "_hip_pitch": 0.0, side + "_knee_pitch": offset,
                      side + "_ankle_pitch": -offset}
            world = np.eye(4)
            for index, body in enumerate(chain):
                local = transform(body)
                if index == 0:
                    local[2, 3] = float(cfg["rewards"]["base_height_target"])
                world = world @ local
                for joint in body.findall("joint"):
                    if joint.get("type", "hinge") != "hinge":
                        continue
                    axis = vector(joint.get("axis"))
                    axis /= np.linalg.norm(axis)
                    angle = angles.get(joint.get("name"), 0.0) - float(joint.get("ref", 0.0))
                    rotation = Rotation.from_rotvec(axis * angle).as_matrix()
                    pivot = vector(joint.get("pos"))
                    hinge = np.eye(4)
                    hinge[:3, :3] = rotation
                    hinge[:3, 3] = pivot - rotation @ pivot
                    world = world @ hinge
            points = vertices @ world[:3, :3].T + world[:3, 3]
            rows.append({"side": side, "hip_pitch_rad": 0.0, "knee_pitch_rad": offset,
                         "ankle_pitch_rad": -offset, "ankle_roll_rad": 0.0,
                         "lowest_sole_height_m": float(points[:, 2].min())})
    return {"method": "Static forward kinematics of the authored V7 MJCF with a level chassis at its reward height. Knee/ankle counter-rotation keeps the sole nearly flat. These are example poses, not a global workspace bound, collision certification, loaded actuator result or walking prediction.",
            "chassis_height_m": cfg["rewards"]["base_height_target"],
            "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest(), "poses": rows}


if __name__ == "__main__":
    target = ROOT / "diagnostics" / "pitch_workspace.json"
    target.write_text(json.dumps(build_report(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Wrote", target)
