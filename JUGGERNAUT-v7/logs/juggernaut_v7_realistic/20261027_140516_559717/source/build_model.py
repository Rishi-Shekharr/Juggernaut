"""Build a simulation-ready JUGGERNAUT v7 MJCF using static STL/XML operations only."""
from __future__ import annotations

import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh


ROOT = Path(__file__).resolve().parent
XML_PATH = ROOT / "JUGGERNAUT-v7.xml"
RAW_XML_PATH = ROOT / "JUGGERNAUT-v7.raw.xml"
CALIBRATION_PATH = ROOT / "component_calibration.json"
DIAGNOSTICS = ROOT / "diagnostics"
MANIFEST_PATH = DIAGNOSTICS / "model_manifest.json"

# Slice profiles are local STL coordinates after scaling to metres. For mirrored
# STLs, reflect the profile's z interval so the cuts follow the native mesh.
HIPFORWARD_SLICE_PROFILE = [((-math.inf, 0.03305, -math.inf), (math.inf, 0.037, math.inf)),
                            ((-math.inf, 0.037, -math.inf), (math.inf, 0.0763, math.inf))]
SLICE_PROFILES = {
    "chas-v7_1": [((-math.inf, -math.inf, -math.inf), (math.inf, -0.023, math.inf)),
                  ((-math.inf, -0.023, -math.inf), (math.inf, 0.023, math.inf)),
                  ((-math.inf, 0.023, -math.inf), (math.inf, math.inf, math.inf))],
    "hipforward-v4_1": HIPFORWARD_SLICE_PROFILE,
    "hipforward-v5_1": HIPFORWARD_SLICE_PROFILE,
    "hipforward-v4Mirror_1": HIPFORWARD_SLICE_PROFILE,
    "servo.MG995-v1_1": [((-0.02025, 0.0, -0.010), (0.02025, 0.0393, 0.010))],
    "servo.MG995-v1_2": [((-0.02025, 0.0, -0.010), (0.02025, 0.0393, 0.010))],
    "servo.MG995-v1Mirror_1": [((-0.02025, 0.0, -0.010), (0.02025, 0.0393, 0.010))],
    "servo.MG995-v1Mirror_2": [((-0.02025, 0.0, -0.010), (0.02025, 0.0393, 0.010))],
    "thigh-v2_1": [((-math.inf, -math.inf, -math.inf), (math.inf, math.inf, math.inf))],
    "thigh-v2Mirror_1": [((-math.inf, -math.inf, -math.inf), (math.inf, math.inf, math.inf))],
    "linky-v2_1": [((-math.inf, -math.inf, -0.044213), (math.inf, math.inf, -0.005))],
    "linky-v2Mirror_1": [((-math.inf, -math.inf, -0.044213), (math.inf, math.inf, -0.005))],
    "MG90S-OnlyBody-v1_1": [((-math.inf, -math.inf, -0.012), (math.inf, math.inf, 0.0161))],
    "MG90S-OnlyBody-v1_2": [((-math.inf, -math.inf, -0.012), (math.inf, math.inf, 0.0161))],
    "MG90S-OnlyBody-v1Mirror_1": [((-math.inf, -math.inf, -0.012), (math.inf, math.inf, 0.0161))],
    "MG90S-OnlyBody-v1Mirror_2": [((-math.inf, -math.inf, -0.012), (math.inf, math.inf, 0.0161))],
    "knee-v3_1": [((-0.02025, -0.0363, -0.030), (0.02025, 0.00695, -0.010)),
                  ((-0.02825, -0.0265, -0.038), (0.02025, 0.00695, -0.030)),
                  ((0.02025, -0.0265, -0.038), (0.05325, -0.0165, -0.010))],
    "knee-v3Mirror_1": [((-0.02025, -0.0363, -0.030), (0.02025, 0.00695, -0.010)),
                        ((-0.02825, -0.0265, -0.038), (0.02025, 0.00695, -0.030)),
                        ((0.02025, -0.0265, -0.038), (0.05325, -0.0165, -0.010))],
    "joiner2-v2_1": [((-math.inf, -0.039, -math.inf), (math.inf, -0.033, math.inf)),
                     ((-math.inf, -0.079, -math.inf), (math.inf, -0.069, math.inf)),
                     ((-math.inf, -0.069, 0.058), (math.inf, -0.039, 0.063))],
    "joiner2-v2Mirror_1": [((-math.inf, -0.039, -math.inf), (math.inf, -0.033, math.inf)),
                           ((-math.inf, -0.079, -math.inf), (math.inf, -0.069, math.inf)),
                           ((-math.inf, -0.069, 0.058), (math.inf, -0.039, 0.063))],
    "chas_second_link-v3_1": [((-math.inf, 0.005, -math.inf), (math.inf, 0.009, math.inf)),
                              ((-math.inf, 0.009, 0.077), (math.inf, 0.021, 0.087)),
                              ((-math.inf, 0.009, -0.000163), (math.inf, 0.021, 0.004837))],
    "chas_second_link-v3Mirror_1": [((-math.inf, 0.005, -math.inf), (math.inf, 0.009, math.inf)),
                                    ((-math.inf, 0.009, 0.077), (math.inf, 0.021, 0.087)),
                                    ((-math.inf, 0.009, -0.000163), (math.inf, 0.021, 0.004837))],
}
MIRROR_Z_BODIES = {name for name in SLICE_PROFILES if "Mirror" in name}
HIP_COLLISION_BODIES = ("hipforward-v5_1", "hipforward-v4Mirror_1")
HIP_COLLISION_TEMPLATE_BY_BODY = {
    "hipforward-v5_1": ("hipforward-v4Mirror_1", "reflect_local_z"),
}
PAIRED_ARM_BODY_GROUPS = {
    "mg90s_shoulders": ("MG90S-OnlyBody-v1_1", "MG90S-OnlyBody-v1Mirror_1"),
    "printed_shoulder_brackets": ("Suporte-Servo-Motor-v1_1", "Suporte-Servo-Motor-v1_2"),
    "third_links": ("chas_third_link-v1_1", "chas_third_link-v1Mirror_1"),
    "sg90_mandibles": ("MG90S-OnlyBody-v1_2", "MG90S-OnlyBody-v1Mirror_2"),
    "printed_mandible_brackets": ("Suporte-Servo-Motor-v1_3", "Suporte-Servo-Motor-v1_4"),
    "second_links": ("chas_second_link-v3_1", "chas_second_link-v3Mirror_1"),
}
BEARING_BODIES = {
    "MR105ZZ-v1_1", "MR105ZZ-v1_2", "MR105ZZ-v1_3",
    "MR105ZZ-v1Mirror_1", "MR105ZZ-v1Mirror_2", "MR105ZZ-v1Mirror_3",
}
SOLE_BODIES = {"loong_rightfootpad-v3_1", "loong_rightfootpad-v2Mirror_1"}
SOLE_AND_POST_GEOMS = {
    "left_sole_collision", "right_sole_collision",
    "left_rear_foot_post_collision", "left_front_foot_post_collision",
    "right_rear_foot_post_collision", "right_front_foot_post_collision",
}
GENERATED_ASSET_PREFIX = "collision_build_"
GENERATED_GEOM_PREFIX = "collision_geom_"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")


def body_map(root):
    return {body.get("name"): body for body in root.iter("body") if body.get("name")}


def resolve_mesh_path(root, source_xml: Path, mesh_name: str) -> tuple[Path, tuple[float, float, float]]:
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    mesh_asset = next((mesh for mesh in root.findall("./asset/mesh")
                       if mesh.get("name") == mesh_name), None)
    if mesh_asset is None or not mesh_asset.get("file"):
        raise ValueError(f"Mesh asset {mesh_name!r} has no source file")
    path = (source_xml.parent / meshdir / mesh_asset.get("file")).resolve()
    scale = tuple(float(x) for x in mesh_asset.get("scale", "1 1 1").split())
    if len(scale) != 3 or not all(math.isfinite(x) and x > 0 for x in scale):
        raise ValueError(f"Invalid scale on mesh asset {mesh_name}")
    return path, scale


def load_scaled_stl(path: Path, scale: tuple[float, float, float]):
    mesh = trimesh.load(path, force="mesh", process=True)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"Could not read triangle mesh {path}")
    mesh.vertices = np.asarray(mesh.vertices, dtype=np.float64) * np.asarray(scale, dtype=np.float64)
    return mesh


def crop_to_box(mesh, lower, upper):
    """Crop an STL mesh by six axis-aligned planes without capping the cuts."""
    sliced = mesh.copy()
    for axis in range(3):
        if math.isfinite(lower[axis]):
            normal = np.zeros(3)
            normal[axis] = 1.0
            origin = np.zeros(3)
            origin[axis] = lower[axis]
            vertices, faces, _ = trimesh.intersections.slice_faces_plane(
                sliced.vertices, sliced.faces, normal, origin
            )
            sliced = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if math.isfinite(upper[axis]):
            normal = np.zeros(3)
            normal[axis] = -1.0
            origin = np.zeros(3)
            origin[axis] = upper[axis]
            vertices, faces, _ = trimesh.intersections.slice_faces_plane(
                sliced.vertices, sliced.faces, normal, origin
            )
            sliced = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        if len(sliced.faces) == 0:
            break
    if len(sliced.faces) == 0:
        raise ValueError(f"CAD slice {lower}..{upper} contains no triangles")
    return sliced


def convex_parts(mesh, profiles, mirror_z=False):
    if mirror_z:
        profiles = [(lower[:2] + (-upper[2],), upper[:2] + (-lower[2],))
                    for lower, upper in profiles]
    parts = []
    for lower, upper in profiles:
        sliced = crop_to_box(mesh, lower, upper)
        hull = sliced.convex_hull
        if len(hull.faces) == 0:
            raise ValueError(f"Convex hull generation returned no faces for slice {lower}..{upper}")
        parts.append(hull)
    parts.sort(key=lambda hull: tuple(np.round(np.r_[hull.bounds.reshape(-1), hull.centroid], 9)))
    return parts


def remove_old_generated_collision(root):
    asset = root.find("asset")
    if asset is not None:
        for mesh in list(asset.findall("mesh")):
            if mesh.get("name", "").startswith(GENERATED_ASSET_PREFIX):
                asset.remove(mesh)
    for element in root.iter():
        for geom in list(element.findall("geom")):
            if geom.get("name", "").startswith(GENERATED_GEOM_PREFIX):
                element.remove(geom)
    contact = root.find("contact")
    if contact is not None:
        for pair in list(contact.findall("pair")):
            if pair.get("name", "").startswith("collision_pair_"):
                contact.remove(pair)


def active_collision(geom) -> bool:
    return geom.get("contype", "1") != "0" and geom.get("conaffinity", "1") != "0"


def clear_nonfoot_colliders(root):
    for body in root.iter("body"):
        for geom in list(body.findall("geom")):
            name = geom.get("name", "")
            if name in SOLE_AND_POST_GEOMS:
                continue
            if active_collision(geom):
                body.remove(geom)


def add_inline_mesh(asset, asset_name: str, mesh):
    vertices = " ".join(f"{float(v):.9g}" for v in np.asarray(mesh.vertices).reshape(-1))
    faces = " ".join(str(int(v)) for v in np.asarray(mesh.faces).reshape(-1))
    ET.SubElement(asset, "mesh", {"name": asset_name, "vertex": vertices, "face": faces})


def visual_mesh_geom(body):
    return [geom for geom in body.findall("geom")
            if (geom.get("type") == "mesh" or geom.get("mesh"))
            and geom.get("group") == "2"]


def reflect_local_z(mesh):
    """Return a copy reflected across local z=0, including consistent face winding."""
    reflected = mesh.copy()
    transform = np.eye(4, dtype=np.float64)
    transform[2, 2] = -1.0
    reflected.apply_transform(transform)
    return reflected


def mirrored_vertex_error_m(reference_mesh, candidate_mesh):
    """Maximum bidirectional nearest-vertex error after local-z reflection."""
    from scipy.spatial import cKDTree

    reflected_vertices = np.asarray(reference_mesh.vertices, dtype=np.float64).copy()
    reflected_vertices[:, 2] *= -1.0
    candidate_vertices = np.asarray(candidate_mesh.vertices, dtype=np.float64)
    ref_to_candidate = cKDTree(candidate_vertices).query(reflected_vertices, k=1)[0]
    candidate_to_ref = cKDTree(reflected_vertices).query(candidate_vertices, k=1)[0]
    return float(max(ref_to_candidate.max(initial=0.0), candidate_to_ref.max(initial=0.0)))


def add_compact_colliders(root, source_xml: Path):
    validate_hip_collision_profiles()
    asset = root.find("asset")
    if asset is None:
        raise ValueError("JUGGERNAUT XML has no asset element")
    bodies = body_map(root)
    cache = {}
    manifest_bodies = {}
    geom_total = 0

    # Derive the right hip hulls from the canonical left hip hulls. Validate the
    # source CAD meshes first so this exact proxy symmetry does not hide real CAD changes.
    hip_templates = {}
    for derived_name, (template_name, transform_name) in HIP_COLLISION_TEMPLATE_BY_BODY.items():
        if transform_name != "reflect_local_z":
            raise ValueError(f"Unsupported hip collision transform {transform_name!r}")
        if derived_name not in bodies or template_name not in bodies:
            raise ValueError(f"Missing hip body for collision template {template_name!r} -> {derived_name!r}")
        template_visuals = visual_mesh_geom(bodies[template_name])
        derived_visuals = visual_mesh_geom(bodies[derived_name])
        if len(template_visuals) != 1 or len(derived_visuals) != 1:
            raise ValueError(f"Expected one visual mesh on paired hip bodies {template_name} and {derived_name}")
        template_mesh_name = template_visuals[0].get("mesh")
        derived_mesh_name = derived_visuals[0].get("mesh")
        template_path, template_scale = resolve_mesh_path(root, source_xml, template_mesh_name)
        derived_path, derived_scale = resolve_mesh_path(root, source_xml, derived_mesh_name)
        if template_scale != derived_scale:
            raise ValueError(f"Paired hip mesh scales differ: {template_scale} vs {derived_scale}")
        template_mesh = load_scaled_stl(template_path, template_scale)
        derived_mesh = load_scaled_stl(derived_path, derived_scale)
        symmetry_error = mirrored_vertex_error_m(template_mesh, derived_mesh)
        if symmetry_error > 1e-7:
            raise ValueError(
                f"Paired hip raw CAD meshes are not local-z mirrors: "
                f"{template_name} vs {derived_name}, max vertex error={symmetry_error:.12g} m"
            )
        template_profiles = SLICE_PROFILES[template_name]
        template_parts = convex_parts(
            template_mesh, template_profiles, template_name in MIRROR_Z_BODIES
        )
        hip_templates[derived_name] = {
            "parts": [reflect_local_z(part) for part in template_parts],
            "raw_mesh": derived_mesh,
            "template_name": template_name,
            "template_mesh_name": template_mesh_name,
            "template_source_hash": sha256(template_path),
            "derived_mesh_name": derived_mesh_name,
            "derived_source_hash": sha256(derived_path),
            "scale": derived_scale,
            "symmetry_error_m": symmetry_error,
            "transform_name": transform_name,
        }

    for name, body in bodies.items():
        if name in BEARING_BODIES or name in SOLE_BODIES:
            continue
        visuals = visual_mesh_geom(body)
        if not visuals:
            continue
        if len(visuals) != 1:
            raise ValueError(f"Expected one body-local visual mesh for {name}, found {len(visuals)}")
        visual = visuals[0]
        mesh_name = visual.get("mesh")
        if name in hip_templates:
            template = hip_templates[name]
            path, scale = resolve_mesh_path(root, source_xml, mesh_name)
            mesh = template["raw_mesh"]
            source_hash = template["derived_source_hash"]
            parts = template["parts"]
            geometry_source_hash = template["template_source_hash"]
            geometry_transform_tag = "_reflect_local_z"
        else:
            path, scale = resolve_mesh_path(root, source_xml, mesh_name)
            source_hash = sha256(path)
            profiles = SLICE_PROFILES.get(name, [((-math.inf, -math.inf, -math.inf),
                                                    (math.inf, math.inf, math.inf))])
            print(f"[collision] {name}: CAD slices={len(profiles)}", flush=True)
            cache_key = (source_hash, scale, tuple(profiles), name in MIRROR_Z_BODIES)
            if cache_key not in cache:
                source_mesh = load_scaled_stl(path, scale)
                parts = convex_parts(source_mesh, profiles, name in MIRROR_Z_BODIES)
                cache[cache_key] = (parts, source_mesh, source_hash)
            parts, mesh, _ = cache[cache_key]
            geometry_source_hash = source_hash
            geometry_transform_tag = ""
        profiles = SLICE_PROFILES.get(name, [((-math.inf, -math.inf, -math.inf),
                                                (math.inf, math.inf, math.inf))])
        print(f"[collision] {name}: achieved hulls={len(parts)}", flush=True)
        body_hulls = []
        for index, hull in enumerate(parts):
            profile_tag = hashlib.sha256(repr(tuple(profiles)).encode()).hexdigest()[:8]
            asset_name = (f"{GENERATED_ASSET_PREFIX}{geometry_source_hash[:12]}"
                          f"_s{hashlib.sha256(','.join(f'{value:.12g}' for value in scale).encode()).hexdigest()[:8]}"
                          f"_p{profile_tag}{geometry_transform_tag}_h{index}")
            if not any(node.get("name") == asset_name for node in asset.findall("mesh")):
                add_inline_mesh(asset, asset_name, hull)
            geom_name = f"{GENERATED_GEOM_PREFIX}{safe_name(name)}_{index}"
            geom_attrib = {
                "name": geom_name, "type": "mesh", "mesh": asset_name,
                "class": "collision", "contype": "1", "conaffinity": "1", "mass": "0",
            }
            for attr in ("pos", "quat", "euler", "axisangle"):
                if attr in visual.attrib:
                    geom_attrib[attr] = visual.get(attr)
            ET.SubElement(body, "geom", geom_attrib)
            body_hulls.append({"asset": asset_name, "vertices": int(len(hull.vertices)),
                               "faces": int(len(hull.faces))})
            geom_total += 1
        body_manifest = {
            "source_mesh": mesh_name,
            "source_sha256": source_hash,
            "scale": list(scale),
            "slice_profile_count": len(profiles),
            "achieved_hulls": len(parts),
            "slice_profiles_local_m": [
                [[value if math.isfinite(value) else None for value in lower],
                 [value if math.isfinite(value) else None for value in upper]]
                for lower, upper in profiles
            ],
            "z_slice_bounds_reflected_for_mirror_mesh": name in MIRROR_Z_BODIES,
            "approximation_note": "Convex hull of each uninflated CAD-plane slice; slice hulls can bridge local openings and are not surface-certified",
            "surface_bounds_m": np.asarray(mesh.bounds).tolist(),
            "hulls": body_hulls,
        }
        if name in hip_templates:
            template = hip_templates[name]
            body_manifest.update({
                "collision_template_mesh": template["template_name"],
                "collision_template_source_mesh": template["template_mesh_name"],
                "collision_template_source_sha256": template["template_source_hash"],
                "collision_template_transform_local": "reflect_local_z",
                "raw_mesh_mirror_max_vertex_error_m": template["symmetry_error_m"],
            })
        manifest_bodies[name] = body_manifest
    return manifest_bodies, geom_total


def update_collision_custom_and_mass(root):
    custom = root.find("custom")
    if custom is None:
        custom = ET.SubElement(root, "custom")
    collision_note = (
        "Compact inline convex collision proxies are generated from original V7 STL surfaces "
        "using deterministic local CAD-plane slices and one convex hull per slice, without "
        "inflation. These are planar-slice approximations, not surface-certified meshes; a hull "
        "may bridge openings within its slice. Six bearings remain visual-only within covered fixed "
        "clusters; each linky support has one central-bar hull cropped away from annular bearing-end "
        "mating regions. The visual/inertial linky bodies remain complete. Both foot soles retain "
        "their named boxes and two post boxes fitted to native V7 plate extents. The robot collision "
        "budget is at most 64 geoms, with 256 "
        "Genesis pair/contact buffers. Self, adjacent, and neutral collision flags are enabled; "
        "MuJoCo contact/constraint solving is enabled with parent filtering disabled, 80 solver "
        "iterations, and 1e-8 tolerance. Original visual STL assets/transforms and body kinematics "
        "remain unchanged. Single-hull proxies can bridge openings; interface clearance is not "
        "certified without simulation. No collision motion tests or model loading were run."
    )
    note = next((item for item in custom.findall("text")
                 if item.get("name") == "collision_model"), None)
    if note is None:
        note = ET.SubElement(custom, "text", {"name": "collision_model"})
    note.set("data", collision_note)
    hardware = next((item for item in custom.findall("text") if item.get("name") == "hardware_assumptions"), None)
    if hardware is None: hardware = ET.SubElement(custom, "text", {"name":"hardware_assumptions"})
    hardware.set("data", "Eight MG996R leg servos: kp=3, kd=0.06 N*m*s/rad, torque cap=0.94 N*m; 6 rad/s DC no-load speed reference. Two MG90S shoulders and two SG90 mandibles (CAD case mesh labels remain MG90S): kp=0.6, kd=0.01808782111111, torque cap=0.1808782111111 N*m; 10 rad/s DC no-load speed reference. Speed references describe the motor envelope; the physical model has no imposed speed limit and external loads may back-drive joints faster. Six leg pitch joints have +/-80 degree physical travel and +/-0.9 rad policy target offsets; leg roll targets remain +/-0.15 rad.")

    total_mass = sum(float(node.get("mass", "0")) for node in root.iter("inertial"))
    mass = next((item for item in custom.findall("numeric")
                 if item.get("name") == "estimated_total_mass_kg"), None)
    if mass is None:
        mass = ET.SubElement(custom, "numeric", {"name": "estimated_total_mass_kg"})
    mass.set("data", f"{total_mass:.13f}")
    return total_mass


def component_assumption(name):
    n = (name or "").lower()
    if n.startswith("chas-v7"):
        return "chassis_battery_electronics_payload", 0.370
    if n.startswith("hipforward-v5"):
        return "right_hip_mg996r_plus_printed_bracket", 0.07640794007572
    if "mg90s-onlybody" in n:
        return ("sg90_mandible_servo" if n.endswith("_2") else "mg90s_shoulder_servo"), (0.009 if n.endswith("_2") else 0.0134)
    if n.startswith("suporte-servo"):
        return "printed_servo_bracket", 0.00515468327615
    if "second_link" in n:
        return "printed_mandible_link", 0.00548698381981
    if "third_link" in n:
        return "printed_arm_link", 0.00215762735851
    if n.startswith("mg996_arm_round"):
        return "mg996r_servo_horn", 0.00683861686585
    if "mr105" in n:
        return "mr105_bearing", 0.00170842911930
    raise ValueError(f"Unmatched V7 component has no documented mass assumption: {name}")


def apply_calibration(root):
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    used, fallback = set(), {}
    for body in root.iter("body"):
        name = body.get("name")
        inertial = body.find("inertial")
        ref = calibration.get(name)
        if ref and inertial is not None:
            inertial.attrib.clear()
            inertial.attrib.update(ref)
            used.add(name)
            # The CAD case on the two mandibles is actually the SG90 hardware.
            if name.startswith("MG90S-OnlyBody"):
                target = 0.009 if name.endswith("_2") or name.endswith("Mirror_2") else 0.0134
                old = float(inertial.get("mass")); ratio = target / old
                vals = [float(x) * ratio for x in inertial.get("fullinertia").split()]
                inertial.set("mass", f"{target:.9g}")
                inertial.set("fullinertia", " ".join(f"{x:.9g}" for x in vals))
        elif inertial is not None:
            component_class, mass = component_assumption(name)
            old = float(inertial.get("mass", str(mass)))
            inertial.set("mass", f"{mass:.12g}")
            try:
                vals = [float(x) for x in inertial.get("fullinertia", "1e-6 1e-6 1e-6 0 0 0").split()]
                if old > 0 and len(vals) == 6:
                    vals = [v * mass / old for v in vals]
                    inertial.set("fullinertia", " ".join(f"{v:.12g}" for v in vals))
            except (ValueError, TypeError):
                inertial.set("fullinertia", "1e-6 1e-6 1e-6 0 0 0")
            fallback[name] = {"component_class":component_class,"assumed_mass_kg":mass,
                "method":"V7 CAD centroid retained; six full-inertia terms scaled by target/raw density mass ratio"}
    return sorted(used), fallback


def validate_paired_arm_masses(root):
    bodies = body_map(root)
    paired_masses = {}
    for group, names in PAIRED_ARM_BODY_GROUPS.items():
        if len(names) != 2 or any(name not in bodies for name in names):
            raise ValueError(f"Expected both bodies for paired arm mass group {group}: {names}")
        masses = [float(bodies[name].find("inertial").get("mass")) for name in names]
        if not math.isclose(masses[0], masses[1], rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"Paired arm masses differ for {group}: {dict(zip(names, masses))}")
        paired_masses[group] = {name: mass for name, mass in zip(names, masses)}
    return paired_masses


def validate_hip_collision_profiles():
    if len(HIP_COLLISION_BODIES) != 2:
        raise ValueError("Expected exactly two paired hip collision proxies")
    profiles = [SLICE_PROFILES.get(name) for name in HIP_COLLISION_BODIES]
    if any(profile is None for profile in profiles) or profiles[0] != profiles[1]:
        raise ValueError(f"Paired hip collision profiles are missing or differ: {HIP_COLLISION_BODIES}")
    if len(profiles[0]) != 2:
        raise ValueError(f"Expected two CAD slices per hip collision proxy, got {len(profiles[0])}")


def configure_dynamics(root):
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("inertiafromgeom", "false")
        compiler.set("balanceinertia", "false")
        compiler.set("autolimits", "true")
    option = root.find("option")
    if option is None: option = ET.SubElement(root, "option")
    option.set("gravity", "0 0 -9.81")
    option.set("timestep", "0.002")
    option.set("integrator", "implicitfast")
    option.set("solver", "Newton")
    option.set("iterations", "80")
    option.set("tolerance", "1e-8")
    flag = option.find("flag")
    if flag is None: flag = ET.SubElement(option, "flag")
    flag.set("contact", "enable"); flag.set("constraint", "enable"); flag.set("filterparent", "disable")
    size = root.find("size")
    if size is None: size = ET.SubElement(root, "size")
    size.set("nconmax", "256"); size.set("njmax", "1000")
    world = root.find("worldbody")
    chassis = next((b for b in world.findall("body") if b.get("name") == "chas-v7_1"), None)
    if chassis is None: raise ValueError("V7 chassis root missing")
    root_xyz = [float(x) for x in chassis.get("pos", "0 0 0").split()]
    root_xyz[2] = 0.171908560947964
    chassis.set("pos", " ".join(f"{x:.15g}" for x in root_xyz))
    if chassis.find("freejoint") is None and not any(j.get("type") == "free" for j in chassis.findall("joint")):
        chassis.insert(0, ET.Element("freejoint", {"name":"floating_base"}))
    joint_ranges = {name: math.radians(75) for name in (
        "right_hip_pitch","right_knee_pitch","right_ankle_pitch","right_ankle_roll",
        "left_hip_pitch","left_knee_pitch","left_ankle_pitch","Revolute-62",
        "right_shoulder","left_shoulder")}
    joint_ranges.update({name: math.radians(80) for name in (
        "right_hip_pitch", "right_knee_pitch", "right_ankle_pitch",
        "left_hip_pitch", "left_knee_pitch", "left_ankle_pitch",
        "right_mandible", "left_mandible")})
    for joint in root.iter("joint"):
        name = joint.get("name", "")
        if joint.get("type") != "hinge": continue
        limit = joint_ranges.get(name, math.radians(75))
        joint.set("limited", "true")
        joint.set("range", f"{-limit:.12g} {limit:.12g}")
        joint.set("damping", "0.02")
        joint.set("armature", "0.00005")
    defaults = root.find("default")
    if defaults is None:
        defaults = ET.Element("default")
        root.insert(list(root).index(root.find("worldbody")), defaults)
    visual = ET.SubElement(defaults, "default", {"class":"visual"})
    ET.SubElement(visual, "geom", {"contype":"0", "conaffinity":"0", "group":"2", "mass":"0"})
    collision = ET.SubElement(defaults, "default", {"class":"collision"})
    ET.SubElement(collision, "geom", {"group":"3", "mass":"0", "condim":"3",
        "friction":"0.8 0.001 0.0001", "solref":"0.008 1", "solimp":"0.95 0.99 0.001"})
    # Native plate extent proxies: sole plus front/rear posts, both colliding with all links.
    for body_name, specs in {
        "loong_rightfootpad-v3_1": [("right_sole_collision",(-0.0125,0.0025,-0.036),(0.0325,0.0025,0.056)),
          ("right_front_foot_post_collision",(-0.0125,0.025,-0.001),(0.0125,0.02,0.009)),
          ("right_rear_foot_post_collision",(-0.0125,0.025,-0.0635),(0.0125,0.02,0.0115))],
        "loong_rightfootpad-v2Mirror_1": [("left_sole_collision",(-0.0125,0.0025,0.036),(0.0325,0.0025,0.056)),
          ("left_front_foot_post_collision",(-0.0125,0.025,0.001),(0.0125,0.02,0.009)),
          ("left_rear_foot_post_collision",(-0.0125,0.025,0.0635),(0.0125,0.02,0.0115))]}.items():
        body = next((b for b in root.iter("body") if b.get("name")==body_name), None)
        if body is None: continue
        for name,pos,size in specs:
            ET.SubElement(body,"geom",{"name":name,"type":"box","class":"collision",
                "pos":" ".join(map(str,pos)),"size":" ".join(map(str,size)),"contype":"1","conaffinity":"1"})
    actuators = root.find("actuator")
    if actuators is not None: root.remove(actuators)
    actuators = ET.SubElement(root, "actuator")
    actuator_order = ["left_hip_pitch","left_knee_pitch","left_ankle_pitch","Revolute-62",
        "right_hip_pitch","right_knee_pitch","right_ankle_pitch","right_ankle_roll",
        "left_shoulder","left_mandible","right_shoulder","right_mandible"]
    jnames = {j.get("name") for j in root.iter("joint")}
    for name in actuator_order:
        if name not in jnames: raise ValueError(f"Missing actuator joint {name}")
        if name in actuator_order[:8]:
            kp, kv, cap = "3", "0.06", "0.94"
        else:
            kp, kv, cap = "0.6", "0.01808782111111", "0.1808782111111"
        limit = joint_ranges[name]
        ET.SubElement(actuators, "position", {"name":name+"_position", "joint":name,
            "kp":kp, "kv":kv, "ctrlrange":f"{-limit:.12g} {limit:.12g}",
            "forcerange":f"-{cap} {cap}", "ctrllimited":"true", "forcelimited":"true"})
    # Flat world floor and chassis-mounted IMU reference frame.
    for geom in list(world.findall("geom")):
        if geom.get("name") == "floor": world.remove(geom)
    ET.SubElement(world,"geom",{"name":"floor","type":"plane","pos":"0 0 0","size":"2 2 0.1",
        "contype":"1","conaffinity":"1","friction":"0.8 0.001 0.0001","condim":"3"})
    for site in list(chassis.findall("site")):
        if site.get("name") == "imu_site": chassis.remove(site)
    ET.SubElement(chassis,"site",{"name":"imu_site","pos":"0 0 0","size":"0.006","rgba":"1 0.2 0.1 1"})
    sensor=root.find("sensor")
    if sensor is not None:root.remove(sensor)
    sensor=ET.SubElement(root,"sensor")
    ET.SubElement(sensor,"gyro",{"name":"imu_gyro","site":"imu_site"})
    ET.SubElement(sensor,"accelerometer",{"name":"imu_accelerometer","site":"imu_site"})
    ET.SubElement(sensor,"framequat",{"name":"imu_orientation","objtype":"site","objname":"imu_site"})
    # Match qpos order from the nested CAD tree, actuator ctrl order follows joints.yaml.
    qpos=["-0.00388449053666498","2.87082651711009e-06","0.171908560947964","0","0","0","1"]
    arm_q={"right_shoulder":math.radians(-60),"left_shoulder":math.radians(-60),
           "right_mandible":math.radians(45),"left_mandible":math.radians(45)}
    for joint in root.iter("joint"):
        if joint.get("type")=="hinge": qpos.append(f"{arm_q.get(joint.get('name'),0.):.12g}")
    ctrl=[arm_q.get(n,0.) for n in actuator_order]
    keyframe=root.find("keyframe")
    if keyframe is not None:root.remove(keyframe)
    keyframe=ET.SubElement(root,"keyframe")
    ET.SubElement(keyframe,"key",{"name":"stand","qpos":" ".join(qpos),"ctrl":" ".join(f"{x:.12g}" for x in ctrl)})


def order_mjcf_sections(root):
    rank = {name:index for index,name in enumerate(("compiler","option","size","visual","statistic","default","extension","custom","asset","worldbody","contact","equality","tendon","actuator","sensor","keyframe","include"))}
    children = list(root)
    children.sort(key=lambda node: rank.get(node.tag, 100))
    root[:] = children


def build():
    if not RAW_XML_PATH.is_file() or not CALIBRATION_PATH.is_file():
        raise FileNotFoundError("Expected V7 raw export and local component calibration data")
    DIAGNOSTICS.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(RAW_XML_PATH)
    root = tree.getroot()
    used_calibration, fallback_assumptions = apply_calibration(root)
    paired_arm_masses = validate_paired_arm_masses(root)
    clear_nonfoot_colliders(root)
    configure_dynamics(root)
    manifest_bodies, generated_geom_count = add_compact_colliders(root, RAW_XML_PATH)
    hip_hull_counts = {name: manifest_bodies[name]["achieved_hulls"] for name in HIP_COLLISION_BODIES}
    if any(count != 2 for count in hip_hull_counts.values()):
        raise ValueError(f"Both hips must produce two collision hulls: {hip_hull_counts}")
    total_mass = update_collision_custom_and_mass(root)
    if generated_geom_count + len(SOLE_AND_POST_GEOMS) > 64:
        raise ValueError(f"Generated collision geoms exceed 64: {generated_geom_count + len(SOLE_AND_POST_GEOMS)}")
    order_mjcf_sections(root)
    ET.indent(tree, space="    ")
    tree.write(XML_PATH, encoding="utf-8", xml_declaration=True)
    mesh_hashes = {item["source_mesh"]: item["source_sha256"] for item in manifest_bodies.values()}
    manifest = {"model":"JUGGERNAUT-v7", "builder":Path(__file__).name,
        "collision_method":"deterministic convex hulls of local CAD-plane slices; no inflation or remeshing",
        "collision_geom_count":generated_geom_count+6,
        "estimated_total_mass_kg":total_mass, "actuator_count":12, "actuator_order":["left_hip_pitch","left_knee_pitch","left_ankle_pitch","Revolute-62","right_hip_pitch","right_knee_pitch","right_ankle_pitch","right_ankle_roll","left_shoulder","left_mandible","right_shoulder","right_mandible"],
        "hardware":{"leg":"MG996R kp=3 kd=0.06 cap=0.94Nm dc_no_load_speed_reference=6_rad_s; no imposed speed limit; external backdrive allowed","upper":"MG90S_shoulders_and_SG90_mandibles kp=0.6 kd=0.01808782111111 cap=0.1808782111111Nm dc_no_load_speed_reference=10_rad_s; no imposed speed limit; external backdrive allowed"},
        "policy_command_limits_rad":{"pitch":0.9,"roll":0.15},
        "physical_joint_travel_deg":{"leg_pitch":[-80,80],"leg_roll":[-75,75],"shoulder":[-75,75],"mandible":[-80,80]},
        "floating_base":"floating_base", "neutral_base_pose":{"xyz_m":[-0.00388449053666498,0.000002870826517,0.171908560947964],"quat_wxyz":[0,0,0,1]},
        "suggested_arm_corridor_deg":{"shoulders":[-65,-55],"mandibles":[35,55],"center_targets":{"shoulder":-60,"mandible":45},"status":"candidate inherited from corrected V6 arm raise convention; V7 full-envelope clearance unverified"}, "calibration_bodies":used_calibration, "fallback_component_assumptions":fallback_assumptions,
        "calibration_source":"V7-local component_calibration.json, corrected component assumptions; V7 transforms retained",
        "paired_arm_masses_kg":paired_arm_masses,
        "hip_collision_proxies":{name:{"slice_profile_count":manifest_bodies[name]["slice_profile_count"],
            "achieved_hulls":manifest_bodies[name]["achieved_hulls"]} for name in HIP_COLLISION_BODIES},
        "source_hashes":{"raw_xml":sha256(RAW_XML_PATH),"calibration":sha256(CALIBRATION_PATH),"meshes":mesh_hashes},
        "bodies":manifest_bodies}
    MANIFEST_PATH.write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Wrote {XML_PATH}; estimated mass {total_mass:.3f} kg; {manifest['collision_geom_count']} collision geoms")


if __name__ == "__main__":
    build()


