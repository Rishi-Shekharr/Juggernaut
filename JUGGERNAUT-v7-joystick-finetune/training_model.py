"""Prepare a robot-only V7 Genesis import without adding sensor hardware.

The canonical XML keeps its standalone floor and native MuJoCo sensors.
At training runtime this temporary import uses Genesis's shared floor and
Python motor control. The complete arm, component masses and inertias remain.
"""
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET


def _geom_mask_defaults(root):
    """Resolve geom collision masks through MuJoCo's nested default classes."""
    classes = {"main": {"contype": "1", "conaffinity": "1"}}
    defaults = root.find("default")
    if defaults is None:
        return classes

    def visit(node, parent_name="main"):
        name = node.get("class", parent_name)
        masks = dict(classes.get(parent_name, classes["main"]))
        geom = node.find("geom")
        if geom is not None:
            masks.update({key: geom.get(key, masks[key]) for key in ("contype", "conaffinity")})
        classes[name] = masks
        for child in node.findall("default"):
            visit(child, name)

    visit(defaults)
    return classes


def validate_collision_coverage(root, env_cfg):
    """Require compatible collision masks, joint-cluster coverage and sole contacts."""
    option = root.find("option")
    flag = option.find("flag") if option is not None else None
    if flag is None or flag.get("filterparent") != "disable":
        raise ValueError("JUGGERNAUT XML must disable MuJoCo parent-child contact filtering")

    defaults = _geom_mask_defaults(root)
    clusters = {}
    active_geom_names = set()

    def walk(body, inherited_class="main", cluster="world"):
        child_class = body.get("childclass", inherited_class)
        if body.findall("joint") or body.findall("freejoint"):
            cluster = body.get("name", "unnamed joint body")
        visual = False
        active = False
        for geom in body.findall("geom"):
            geom_class = geom.get("class", child_class)
            masks = defaults.get(geom_class, defaults["main"])
            contype = int(geom.get("contype", masks["contype"]))
            affinity = int(geom.get("conaffinity", masks["conaffinity"]))
            mesh_geom = geom.get("type") == "mesh" or geom.get("mesh") is not None
            visual |= mesh_geom and (
                geom_class == "visual" or geom.get("group") == "2"
                or contype == 0 or affinity == 0
            )
            if contype != 0 and affinity != 0:
                active = True
                if geom.get("name"):
                    active_geom_names.add(geom.get("name"))
        entry = clusters.setdefault(cluster, {"visual": False, "active": False})
        entry["visual"] |= visual
        entry["active"] |= active
        for child in body.findall("body"):
            walk(child, child_class, cluster)

    world = root.find("worldbody")
    if world is None:
        raise ValueError("JUGGERNAUT model has no worldbody")
    for body in world.findall("body"):
        walk(body)

    uncovered = [name for name, state in clusters.items() if state["visual"] and not state["active"]]
    if uncovered:
        raise ValueError("Visual rigid cluster(s) lack active collision geometry: " + ", ".join(uncovered))
    required_soles = set(env_cfg.get("toe_collision_geoms", ()))
    missing_soles = sorted(required_soles - active_geom_names)
    if missing_soles:
        raise ValueError("Required left/right sole collision geom(s) missing or inactive: " + ", ".join(missing_soles))


def prepare_training_mjcf(source, env_cfg):
    if env_cfg.get("foot_switch", {}).get("enabled", False):
        raise ValueError("JUGGERNAUT V7 has no foot contact switches")
    source = Path(source).resolve()
    root = ET.parse(source).getroot()
    validate_collision_coverage(root, env_cfg)
    compiler = root.find("compiler")
    meshdir = compiler.get("meshdir", "") if compiler is not None else ""
    texturedir = compiler.get("texturedir", "") if compiler is not None else ""
    for kind, directory in (("mesh", meshdir), ("texture", texturedir)):
        for asset in root.findall("./asset/" + kind):
            if asset.get("file"):
                asset.set("file", str((source.parent / directory / asset.get("file")).resolve()))
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)
    world = root.find("worldbody")
    if world is None:
        raise ValueError("JUGGERNAUT model has no worldbody")
    base_name = env_cfg["base_link"]
    chassis = world.find("body[@name='" + base_name + "']")
    if chassis is None:
        raise ValueError("Expected JUGGERNAUT chassis " + base_name)
    # One environment floor, no duplicate XML plane or inferred switch housings.
    for element in list(world):
        if element is not chassis:
            world.remove(element)
    for parent in chassis.iter():
        for camera in list(parent.findall("camera")):
            parent.remove(camera)
    # Native control and sensors are not imported as deployable policy inputs.
    for tag in ("actuator", "sensor", "keyframe", "custom"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    temporary = tempfile.TemporaryDirectory(prefix="juggernaut_v7_training_")
    output = Path(temporary.name) / "JUGGERNAUT-v7-training.xml"
    ET.indent(root, space="    ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    return temporary, str(output)
