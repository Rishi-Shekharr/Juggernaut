"""Genesis presentation scene, camera presets and read-only policy telemetry."""
import math

import genesis as gs
import numpy as np
from genesis.vis.viewer_plugins import ViewerPlugin


DEFAULTS = {
    "resolution": [1280, 720], "record_resolution": [1920, 1080],
    "camera_view": "three_quarter", "camera_fov": 38.0,
    "camera_smoothing_s": 0.10, "refresh_rate": 60,
    "background_color": [0.15, 0.25, 0.35], "ambient_light": [0.3, 0.3, 0.3],
    "shadows": True, "floor_reflection": True,
    "guides": True, "telemetry": True,
}
VIEWS = {
    "three_quarter": np.array([0.58, -0.72, 0.18]),
    "side": np.array([0.02, -0.82, 0.17]),
    "top": np.array([0.02, -0.01, 0.90]),
    "rear": np.array([-0.82, 0.0, 0.17]),
}


def settings(cfg):
    result = {**DEFAULTS, **cfg.get("viewer", {})}
    if result["camera_view"] not in VIEWS:
        raise ValueError("viewer.camera_view must be three_quarter, side, top or rear")
    if float(result["camera_smoothing_s"]) < 0:
        raise ValueError("viewer.camera_smoothing_s must be nonnegative")
    return result


def options(cfg, show_viewer):
    view = settings(cfg)
    return gs.options.ViewerOptions(
        res=tuple(view["resolution"]), refresh_rate=view["refresh_rate"],
        camera_pos=(0.58, -0.72, 0.38), camera_lookat=(0.0, 0.0, 0.125),
        camera_fov=view["camera_fov"], enable_help_text=False,
        realtime_factor=1.0 if show_viewer else None,
    ), gs.options.VisOptions(
        background_color=tuple(view["background_color"]),
        ambient_light=tuple(view["ambient_light"]),
        shadow=view["shadows"], plane_reflection=view["floor_reflection"],
        rendered_envs_idx=[0],
        lights=[
            {"type": "directional", "dir": [-0.3, -0.4, -1.0],
             "color": [1.0, 0.96, 0.90], "intensity": 3.0},
            {"type": "directional", "dir": [0.5, 0.2, -0.5],
             "color": [0.65, 0.78, 1.0], "intensity": 1.0},
        ],
    )


def floor_surface():
    # Texturing affects rendering only; collision geometry and friction are separate.
    y, x = np.indices((256, 256))
    check = ((x // 64 + y // 64) % 2).astype(bool)
    image = np.empty((256, 256, 3), dtype=np.uint8)
    image[check] = (51, 77, 102)
    image[~check] = (26, 51, 77)
    image[(x % 64 == 0) | (y % 64 == 0)] = (125, 145, 164)
    return gs.surfaces.Rough(diffuse_texture=gs.textures.ImageTexture(image_array=image))


class TelemetryCaption(ViewerPlugin):
    """Publish captions through the viewer's existing text renderer."""
    def __init__(self, presentation):
        super().__init__()
        self.presentation = presentation

    def build(self, viewer, camera, scene):
        super().build(viewer, camera, scene)
        self.viewer.set_caption("JUGGERNAUT V7 | 12 actuators")

    def on_draw(self):
        from genesis.ext.pyrender.constants import TextAlign
        data = self.presentation.telemetry
        if not data:
            return
        yaw_text = f"sim yaw {data['yaw_deg']:+.1f} deg"
        captions = (
            (TextAlign.TOP_LEFT, f"JUGGERNAUT V7 / 12 ACTUATORS  |  {data['stage']}  |  {data['view']}"),
            (TextAlign.TOP_RIGHT,
             f"t {data['seconds']:.1f}s  |  vx {data['vx']:.3f} / {data['command']:.3f} m/s"),
            (TextAlign.BOTTOM_LEFT,
             f"Sole clearance  L {data['left_mm']:.1f}  R {data['right_mm']:.1f} mm"
             f"  |  {yaw_text}  |  drift {data['drift']:+.3f} m"),
            (TextAlign.BOTTOM_RIGHT, "1 three-quarter   2 side   3 top   4 rear   T follow"),
        )
        self.viewer.viewer_flags["caption"] = [
            {"text": text, "location": location, "font_name": "OpenSans-Regular",
             "font_pt": 18, "color": (0.9, 0.95, 1.0, 1.0), "scale": 1.0}
            for location, text in captions
        ]

    def on_key_press(self, symbol, modifiers):
        from pyglet.window import key
        views = {key._1: "three_quarter", key._2: "side", key._3: "top", key._4: "rear"}
        if symbol in views:
            self.presentation.view_name = views[symbol]
            self.presentation.follow = True
            return True
        if symbol == key.T:
            self.presentation.follow = not self.presentation.follow
            return True
        return None


class PolicyPresentation:
    def __init__(self, env, show_viewer):
        self.env = env
        self.show_viewer = show_viewer
        self.cfg = settings(env.cfg)
        self.view_name = self.cfg["camera_view"]
        self.follow = True
        self.camera_pos = None
        self.camera_lookat = None
        self.telemetry = {}
        self.elapsed = 0.0
        self.guide_nodes = []
        if show_viewer and self.cfg["telemetry"]:
            env.scene.viewer.add_plugin(TelemetryCaption(self))
        if self.cfg["guides"]:
            for lateral, color in ((0.0, (0.5, 0.8, 1.0, 0.7)),
                                   (-0.1, (0.6, 0.7, 0.8, 0.4)), (0.1, (0.6, 0.7, 0.8, 0.4))):
                env.scene.draw_debug_line((-0.5, lateral, 0.001), (10.0, lateral, 0.001),
                                          radius=0.0007, color=color)
        self.update(advance_time=False)

    def update(self, advance_time=True):
        env = self.env
        if advance_time:
            self.elapsed += env.dt
        position = np.asarray(env.base_pos[0].detach().cpu().tolist())
        clearance = env.foot_clearance[0].clamp_min(0).detach().cpu().tolist()
        velocity = env._command_velocity()[0, 0].item()
        self.telemetry = {
            "stage": env.cfg.get("training", {}).get("stage", env.servo_mode),
            "view": self.view_name, "seconds": self.elapsed,
            "vx": velocity, "command": env.commands[0, 0].item(),
            "left_mm": clearance[0] * 1000, "right_mm": clearance[1] * 1000,
            "yaw_deg": math.degrees(env.heading_error[0].item()),
            "drift": position[1] - env.lateral_origin[0].item(),
        }
        if self.follow:
            offset = VIEWS[self.view_name].copy()
            if self.view_name == "rear":
                yaw = env.raw_yaw[0].item()
                c, s = math.cos(yaw), math.sin(yaw)
                offset[:2] = (c * offset[0] - s * offset[1], s * offset[0] + c * offset[1])
            target_pos = position + offset
            target_lookat = position + np.array([0.0, 0.0, -0.07])
            tau = float(self.cfg["camera_smoothing_s"])
            alpha = 1.0 if tau == 0 else -math.expm1(-env.dt / tau)
            if self.camera_pos is None:
                self.camera_pos, self.camera_lookat = target_pos, target_lookat
            else:
                self.camera_pos += alpha * (target_pos - self.camera_pos)
                self.camera_lookat += alpha * (target_lookat - self.camera_lookat)
            if self.show_viewer:
                env.scene.viewer.set_camera_pose(pos=self.camera_pos, lookat=self.camera_lookat)
            if env.cam is not None:
                env.cam.set_pose(pos=self.camera_pos, lookat=self.camera_lookat)
        elif self.show_viewer and env.cam is not None:
            env.cam.set_pose(transform=env.scene.viewer.camera_pose)
        if self.cfg["guides"] and (not advance_time or round(self.elapsed / env.dt) % 5 == 0):
            for node in self.guide_nodes:
                env.scene.clear_debug_object(node)
            anchor = position + np.array([0.0, 0.0, 0.075])
            heading_feedback = env.command_reference_frame == "reset_heading"
            if heading_feedback and hasattr(env, "heading_reference"):
                yaw = env.heading_reference[0].item()
            else:
                yaw = env.raw_yaw[0].item()  # legacy commands are expressed in the current body frame
            command = env.commands[0, 0].item()
            world_velocity = env.world_lin_vel[0].detach().cpu().numpy()
            self.guide_nodes = []
            for vector, color in (
                (np.array([math.cos(yaw), math.sin(yaw), 0.0]) * command, (0.95, 0.8, 0.3, 0.95)),
                (np.array([world_velocity[0], world_velocity[1], 0.0]), (0.2, 0.85, 0.95, 0.95)),
            ):
                if np.linalg.norm(vector) > 1e-5:
                    self.guide_nodes.append(env.scene.draw_debug_arrow(anchor, vector,
                                                                       radius=0.0015, color=color))
