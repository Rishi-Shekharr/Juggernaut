"""JUGGERNAUT V7 mixed-position walking physics and fused servo execution.

No project code is executed while preparing these files. Runtime guards below
protect the sensor, timing and equal-clearance contract before training begins.
"""
import math
import os

from sensors import SixAxisIMU
from training_model import prepare_training_mjcf
from external_arms import ExternalArmCommands, validate_external_arms
from stages import REWARD_VERSION

import torch
from torch.overrides import TorchFunctionMode
import genesis as gs



def quat_rotate_inverse(q, v):
    """Rotate vectors v by the inverse of unit quaternions q (w, x, y, z order)."""
    w = q[..., :1]
    xyz = q[..., 1:]
    a = v * (2.0 * w * w - 1.0)
    b = torch.cross(xyz, v, dim=-1) * w * 2.0
    c = xyz * (xyz * v).sum(dim=-1, keepdim=True) * 2.0
    return a - b + c


def rand_uniform(lo, hi, shape, device):
    return torch.rand(shape, device=device, dtype=gs.tc_float) * (hi - lo) + lo


EXPECTED_JOINTS = (
    ("left_hip_pitch", "hip_pitch"),
    ("left_knee_pitch", "knee"),
    ("left_ankle_pitch", "ankle_pitch"),
    ("Revolute-62", "ankle_roll"),
    ("right_hip_pitch", "hip_pitch"),
    ("right_knee_pitch", "knee"),
    ("right_ankle_pitch", "ankle_pitch"),
    ("right_ankle_roll", "ankle_roll"),
    ("left_shoulder", "shoulder"),
    ("left_mandible", "mandible"),
    ("right_shoulder", "shoulder"),
    ("right_mandible", "mandible"),
)
LEG_JOINT_COUNT = 8
ARM_JOINT_COUNT = 4
PITCH_ACTION_SCALE = 0.9
ROLL_ACTION_SCALE = 0.15
PITCH_TARGET_LIMIT_RAD = 1.3962634015955
V7_PIPELINE = "juggernaut_v7_joystick_v1"
LEG_SPEED_CAP_RAD_S = 6.0
LEG_PEAK_TORQUE_NM = 0.94
ARM_SPEED_CAP_RAD_S = 10.0
MANDIBLE_SPEED_CAP_RAD_S = ARM_SPEED_CAP_RAD_S
ARM_PEAK_TORQUE_NM = 0.1808782111111
MANDIBLE_PEAK_TORQUE_NM = ARM_PEAK_TORQUE_NM


def validate_joint_config(joints_cfg):
    """Require eight policy leg joints followed by four externally controlled motors."""
    joints = joints_cfg.get("joints")
    actual = tuple((joint.get("name"), joint.get("role")) for joint in joints or ())
    if actual != EXPECTED_JOINTS:
        names = ", ".join(name for name, _ in EXPECTED_JOINTS)
        raise ValueError(f"Expected exactly twelve actuated joints in this order: {names}")
    defaults = joints_cfg.get("defaults", {})
    for joint in joints or ():
        name, role = joint.get("name"), joint.get("role")
        scale = float(joint.get("action_scale", defaults.get("action_scale", 0.0)))
        if role in ("hip_pitch", "knee", "ankle_pitch"):
            bounds = joint.get("target_limit", joints_cfg.get("target_limit", ()))
            if (not math.isclose(scale, PITCH_ACTION_SCALE, rel_tol=0, abs_tol=1e-12)
                    or len(bounds) != 2
                    or not math.isclose(float(bounds[0]), -PITCH_TARGET_LIMIT_RAD, rel_tol=0, abs_tol=1e-12)
                    or not math.isclose(float(bounds[1]), PITCH_TARGET_LIMIT_RAD, rel_tol=0, abs_tol=1e-12)):
                raise ValueError(f"{name} requires action_scale=0.9 and target limits plus/minus 80 degrees for the V7 v4 action contract")
        elif role == "ankle_roll" and not math.isclose(scale, ROLL_ACTION_SCALE, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{name} requires action_scale=0.15 for the V7 v4 action contract")


REALISTIC_SERVO_SPEED_CAP_RAD_S = ARM_SPEED_CAP_RAD_S


def validate_realistic_servo_limits(cfg, joints_cfg):
    """Validate leg and MG90S limits against their separate hardware caps."""
    servo = cfg.get("env", {}).get("servo", {})
    if servo.get("mode", "pd") != "torque_speed":
        return

    def bounded_positive(value, label, maximum):
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be finite, positive, and no greater than {maximum:g} rad/s")
        if not math.isfinite(value) or value <= 0 or value > maximum:
            raise ValueError(f"{label} must be finite, positive, and no greater than {maximum:g} rad/s")
        return value

    bounded_positive(servo.get("target_slew_speed_rad_s"), "servo.target_slew_speed_rad_s", LEG_SPEED_CAP_RAD_S)
    bounded_positive(servo.get("hard_velocity_limit_rad_s"), "servo.hard_velocity_limit_rad_s", LEG_SPEED_CAP_RAD_S)
    bounded_positive(servo.get("arm_target_slew_speed_rad_s"), "servo.arm_target_slew_speed_rad_s", ARM_SPEED_CAP_RAD_S)
    bounded_positive(servo.get("arm_hard_velocity_limit_rad_s"), "servo.arm_hard_velocity_limit_rad_s", ARM_SPEED_CAP_RAD_S)
    defaults = joints_cfg.get("defaults", {})
    for joint in joints_cfg.get("joints", ()):
        role = joint.get("role")
        cap = ARM_SPEED_CAP_RAD_S if role in {role_name for _, role_name in EXPECTED_JOINTS[LEG_JOINT_COUNT:]} else LEG_SPEED_CAP_RAD_S
        speed = joint.get("max_speed", defaults.get("max_speed"))
        bounded_positive(speed, f"max_speed for {joint.get('name', 'joint')}", cap)

    dr = cfg.get("env", {}).get("domain_randomization", {})
    if dr.get("enabled", False):
        scale = dr.get("speed_scale", [1.0, 1.0])
        if not isinstance(scale, (list, tuple)) or len(scale) != 2:
            raise ValueError("domain_randomization.speed_scale must be an ordered two-value range within (0, 1]")
        try:
            lo, hi = (float(value) for value in scale)
        except (TypeError, ValueError):
            raise ValueError("domain_randomization.speed_scale must be an ordered two-value range within (0, 1]")
        if not math.isfinite(lo) or not math.isfinite(hi) or not 0 < lo <= hi <= 1:
            raise ValueError("domain_randomization.speed_scale must be an ordered two-value range within (0, 1]")
        torque_scale = dr.get("torque_scale", [1.0, 1.0])
        if not isinstance(torque_scale, (list, tuple)) or len(torque_scale) != 2:
            raise ValueError("domain_randomization.torque_scale must be an ordered two-value range within (0, 1]")
        try:
            torque_lo, torque_hi = (float(value) for value in torque_scale)
        except (TypeError, ValueError):
            raise ValueError("domain_randomization.torque_scale must be an ordered two-value range within (0, 1]")
        if not math.isfinite(torque_lo) or not math.isfinite(torque_hi) or not 0 < torque_lo <= torque_hi <= 1:
            raise ValueError("domain_randomization.torque_scale must be an ordered two-value range within (0, 1]")


def validate_training_contract(cfg, joints_cfg, *, nominal_speed_rad_s=LEG_SPEED_CAP_RAD_S, peak_torque_nm=LEG_PEAK_TORQUE_NM):
    """Protect V7 sensor isolation, timing, actuator caps and bilateral gait."""
    e, gait = cfg["env"], cfg["gait"]
    if not math.isclose(float(e.get("clip_actions", 0.0)), 1.0, rel_tol=0, abs_tol=1e-10):
        raise ValueError("V7 operating geometry requires clip_actions=1")
    if not 0.0 <= float(e.get("reset_dof_noise", 0.0)) <= 0.02:
        raise ValueError("V7 reset noise must remain within the 0.02 rad CAD operating allowance")
    for key in ("self_collision", "adjacent_collision", "neutral_collision"):
        if not e.get(key, False):
            raise ValueError(f"env.{key} must remain enabled for complete robot collision coverage")
    for key in ("max_collision_pairs", "max_contacts"):
        try:
            capacity = int(e.get(key, 0))
        except (TypeError, ValueError):
            raise ValueError(f"env.{key} must be at least 256")
        if capacity < 256:
            raise ValueError(f"env.{key} must be at least 256")
    if cfg.get("training", {}).get("pipeline_version") != V7_PIPELINE:
        raise ValueError("This environment requires the V7 v4 external-arms history and reward contract; start fresh with a v4 checkpoint")
    if cfg.get("training", {}).get("reward_version") != REWARD_VERSION:
        raise ValueError("This environment requires the " + REWARD_VERSION + " reward definition")
    sigma = float(cfg["rewards"]["tracking_sigma_lin"])
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("rewards.tracking_sigma_lin must be finite and positive")
    validate_joint_config(joints_cfg)
    validate_external_arms(cfg, joints_cfg)
    nominal_leg = e.get("nominal_leg_actuator", {})
    for key, cap in (("speed_rad_s", LEG_SPEED_CAP_RAD_S), ("torque_limit_nm", LEG_PEAK_TORQUE_NM)):
        try:
            value = float(nominal_leg[key])
        except (KeyError, TypeError, ValueError):
            raise ValueError(f"env.nominal_leg_actuator.{key} must be finite, positive, and no greater than its rated hardware cap")
        if not math.isfinite(value) or value <= 0 or value > cap:
            raise ValueError(f"env.nominal_leg_actuator.{key} must be finite, positive, and no greater than {cap:g}")
    dr = e.get("domain_randomization", {})
    defaults = joints_cfg.get("defaults", {})
    if any(float(joint.get("action_scale", defaults.get("action_scale", 0.0))) != 0.0
           for joint in joints_cfg["joints"][LEG_JOINT_COUNT:]):
        raise ValueError("Externally controlled shoulder/mandible action scales must be zero")
    if e.get("ecompass", {}).get("enabled", False) or e.get("heading_feedback", False):
        raise ValueError("JUGGERNAUT has no magnetic heading/actor heading feedback")
    if not e.get("imu", {}).get("enabled", False):
        raise ValueError("V7 requires the six-axis IMU")
    if e.get("foot_switch", {}).get("enabled", False):
        raise ValueError("JUGGERNAUT V7 has no foot contact switches")
    if not e.get("ground_support_only", False) or not e.get("critic_lateral_position", False):
        raise ValueError("Floor-only contacts and privileged lateral position must stay enabled")
    if e.get("skip_command_observer", False) or e.get("skip_training_metrics", False):
        raise ValueError("Command observer and bilateral training metrics must remain active")
    if e.get("joint_velocity_observation") != "command_history":
        raise ValueError("All actor joint-velocity slots must come from the commanded target observer")
    if not e.get("disable_state_projection", False):
        raise ValueError("Physical joint positions and velocities must remain unprojected after physics; servo speed is a motor-envelope reference")
    if any(float(v) != 0 for v in cfg["commands"]["lin_vel_y"]):
        raise ValueError("Lateral joystick translation is not supported")
    if e.get("command_reference_frame") != "body":
        raise ValueError("Joystick linear velocity must be in the robot body frame")
    if abs(float(e["control_dt"])-0.02) > 1e-12 or int(e["substeps"]) != 10:
        raise ValueError("Keep 4dof timing: 20 ms control, ten 2 ms physics substeps")
    if e.get("joint_observation") != "mixed":
        raise ValueError("joint_observation must be mixed for V7")
    sensor = e.get("joint_sensor", {})
    leg_names = [name for name, _ in EXPECTED_JOINTS[:LEG_JOINT_COUNT]]
    arm_names = [name for name, _ in EXPECTED_JOINTS[LEG_JOINT_COUNT:]]
    if sensor.get("measured_joints") != leg_names:
        raise ValueError("joint_sensor.measured_joints must list the eight leg potentiometers in action order")
    if sensor.get("command_estimated_joints") != arm_names:
        raise ValueError("joint_sensor.command_estimated_joints must list the four externally controlled upper-body joints in motor order")
    for key in ("position_noise_std_rad", "position_noise_uniform_rad", "quantization_rad"):
        value = float(sensor.get(key, 0.0))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"joint_sensor.{key} must be finite and nonnegative")
    if e.get("servo", {}).get("mode") != "torque_speed":
        raise ValueError("V7 requires the torque/speed servo")
    servo = e["servo"]
    for key, expected in (("target_slew_speed_rad_s", LEG_SPEED_CAP_RAD_S), ("hard_velocity_limit_rad_s", LEG_SPEED_CAP_RAD_S),
                          ("arm_target_slew_speed_rad_s", ARM_SPEED_CAP_RAD_S), ("arm_hard_velocity_limit_rad_s", ARM_SPEED_CAP_RAD_S),
                          ("mandible_target_slew_speed_rad_s", MANDIBLE_SPEED_CAP_RAD_S), ("mandible_hard_velocity_limit_rad_s", MANDIBLE_SPEED_CAP_RAD_S)):
        if not math.isclose(float(servo.get(key, 0)), expected, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Servo contract requires {key}={expected:g} rad/s")
    backlash = float(servo.get("backlash_rad", 0.0))
    if not math.isfinite(backlash) or backlash < 0:
        raise ValueError("servo.backlash_rad must be a finite nonnegative total play width")
    dr = e.get("domain_randomization", {})
    if dr.get("enabled", False):
        bounds = dr.get("backlash_rad", [backlash, backlash])
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError("domain_randomization.backlash_rad requires two ordered bounds")
        lo, hi = (float(value) for value in bounds)
        if not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo <= hi:
            raise ValueError("domain_randomization.backlash_rad must be finite and nonnegative")
    defaults = joints_cfg["defaults"]
    for index, joint in enumerate(joints_cfg["joints"]):
        arm = index >= LEG_JOINT_COUNT
        speed = float(joint.get("max_speed", defaults.get("max_speed")))
        role = joint.get("role")
        cap = MANDIBLE_SPEED_CAP_RAD_S if role == "mandible" else (ARM_SPEED_CAP_RAD_S if role == "shoulder" else LEG_SPEED_CAP_RAD_S)
        if not math.isclose(speed, cap, rel_tol=0, abs_tol=1e-10):
            raise ValueError(f"Nominal {joint['name']} speed must remain {cap:g} rad/s")
        torque = joint.get("torque_limit", defaults.get("torque_limit"))
        role = joint.get("role")
        torque_cap = MANDIBLE_PEAK_TORQUE_NM if role == "mandible" else (ARM_PEAK_TORQUE_NM if arm else peak_torque_nm)
        if torque is None or not 0 < float(torque) <= torque_cap:
            raise ValueError(f"{joint['name']} peak torque must be <= {torque_cap:g} Nm")
        if arm and not math.isclose(float(joint.get("kp", defaults.get("kp", 0))), 0.6, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Arm kp must remain 0.6")
        if arm and not math.isclose(float(joint.get("kd", defaults.get("kd", 0))), 0.01808782111111, rel_tol=0, abs_tol=1e-12):
            raise ValueError("Arm kd must remain 0.01808782111111")
    safety = e.get("servo_safety", {})
    stall_torque_fraction = float(safety.get("stall_torque_fraction", 0.7))
    stall_speed_rad_s = float(safety.get("stall_speed_rad_s", 0.2))
    max_stall_duration_s = float(safety.get("max_stall_duration_s", 0.5))
    if not math.isfinite(stall_torque_fraction) or not 0 < stall_torque_fraction <= 1:
        raise ValueError("servo_safety.stall_torque_fraction must be in (0, 1]")
    if not math.isfinite(stall_speed_rad_s) or stall_speed_rad_s < 0:
        raise ValueError("servo_safety.stall_speed_rad_s must be finite and nonnegative")
    if not math.isfinite(max_stall_duration_s) or max_stall_duration_s <= 0:
        raise ValueError("servo_safety.max_stall_duration_s must be finite and positive")
    if gait.get("swing_height_mode", "phase") != "phase":
        raise ValueError("Both legs require the same phase-shaped clearance target")
    height, minimum = float(gait["swing_height"]), float(gait["minimum_swing_clearance_m"])
    if not math.isfinite(height) or not math.isfinite(minimum) or not 0 < minimum <= height:
        raise ValueError("A shared nonzero swing height and minimum clearance are required")
    if any(key in gait for key in ("left_swing_height", "right_swing_height", "swing_height_left", "swing_height_right")):
        raise ValueError("Per-leg clearance targets are forbidden; use the single shared swing_height")
    for name in ("feet_swing_height", "feet_swing_peak", "feet_swing_symmetry", "feet_swing_clearance"):
        weight = float(cfg["rewards"]["scales"].get(name, 0))
        if not math.isfinite(weight) or weight >= 0:
            raise ValueError(name+" must remain a negative penalty from the first bootstrap rollout")
    contract = cfg.get("training_contract", {})
    if contract.get("actor_observations") != 230 or contract.get("critic_observations") != 53:
        raise ValueError("JUGGERNAUT v7 observation contract is actor 230 / critic 53 (5 x 46 history)")
    if contract.get("num_actions") != LEG_JOINT_COUNT or contract.get("action_joints") != leg_names:
        raise ValueError("Only the eight leg joints may appear in the policy action contract")
    expected_fields = ["gyro_xyz", "projected_gravity_xyz", "commands_forward_yaw", "measured_leg_joint_positions",
                       "command_estimated_arm_joint_positions", "command_history_joint_velocities",
                       "previous_leg_actions", "external_arm_command_positions", "gait_sin_cos"]
    if contract.get("actor_fields") != expected_fields:
        raise ValueError("Actor fields must include eight leg actions and four explicit external arm commands")
    if (contract.get("frame_observations") != 46 or contract.get("history_frames") != 5
            or contract.get("history_order") != "oldest_to_newest"):
        raise ValueError("V7 requires five 46-value actor frames ordered oldest to newest")
    if e.get("observation_history_frames", 5) != 5:
        raise ValueError("env.observation_history_frames must be five")
    if cfg.get("training", {}).get("num_actions") != LEG_JOINT_COUNT:
        raise ValueError("training.num_actions must be eight; shoulder/mandible actions are forbidden")
    if cfg.get("performance", {}).get("backend", "auto") not in ("auto", "warp", "torch"):
        raise ValueError("performance.backend must be auto, warp or torch")


def validate_curriculum_configuration(cfg):
    """Validate scalar curriculum and randomization ranges before simulation starts."""
    training = cfg.get("training", {})
    curriculum = training.get("curriculum", {})
    if curriculum.get("enabled", False):
        for key, minimum in (("nominal_iterations", 0), ("ramp_iterations", 1)):
            value = curriculum.get(key, 100 if key == "nominal_iterations" else 1500)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"training.curriculum.{key} must be an integer >= {minimum}")
        replay = float(curriculum.get("nominal_replay_probability", 0.0))
        if not math.isfinite(replay) or not 0.0 <= replay <= 1.0:
            raise ValueError("training.curriculum.nominal_replay_probability must be in [0, 1]")

    env = cfg.get("env", {})
    dr = env.get("domain_randomization", {})
    if not dr.get("enabled", False):
        return
    for key in ("kp_scale", "kd_scale", "torque_scale", "speed_scale", "friction_scale", "mass_scale"):
        if key not in dr:
            continue
        bounds = dr[key]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"domain_randomization.{key} requires two ordered positive bounds")
        try:
            lo, hi = (float(value) for value in bounds)
        except (TypeError, ValueError):
            raise ValueError(f"domain_randomization.{key} requires two ordered positive bounds")
        if not math.isfinite(lo) or not math.isfinite(hi) or not 0 < lo <= hi:
            raise ValueError(f"domain_randomization.{key} requires two ordered positive bounds")
        if key in ("torque_scale", "speed_scale") and hi > 1.0:
            raise ValueError(f"domain_randomization.{key} must not raise hardware limits")
    for key in ("deadband_rad", "backlash_rad"):
        if key not in dr:
            continue
        bounds = dr[key]
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"domain_randomization.{key} requires two ordered nonnegative bounds")
        try:
            lo, hi = (float(value) for value in bounds)
        except (TypeError, ValueError):
            raise ValueError(f"domain_randomization.{key} requires two ordered nonnegative bounds")
        if not math.isfinite(lo) or not math.isfinite(hi) or not 0 <= lo <= hi:
            raise ValueError(f"domain_randomization.{key} requires two ordered nonnegative bounds")


class _ContactGatherIndexCompat(TorchFunctionMode):
    """Cast only gather int32 indices inside one public contact-query scope."""
    def __init__(self):
        self.indices = {}

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else dict(kwargs)
        if func in (torch.Tensor.gather, torch.gather):
            index = kwargs.get("index", args[2] if len(args) > 2 else None)
            if index is not None and index.dtype == torch.int32:
                key = id(index)
                if key not in self.indices:
                    # Retain the source too, preventing id reuse during this scope.
                    self.indices[key] = (index, index.long())
                converted = self.indices[key][1]
                if "index" in kwargs:
                    kwargs["index"] = converted
                else:
                    args = (*args[:2], converted, *args[3:])
        return func(*args, **kwargs)


class JuggernautEnv:
    """JUGGERNAUT walking physics, with a 230-input actor / 53-input critic.

    Each actor frame contains a delayed six-axis IMU, XY command, measured leg-potentiometer positions,
    command-observed arm positions and joint velocities, previous leg actions,
    explicit externally commanded arm positions, and gait clock. Five frames are
    stacked oldest-to-newest for the deployable policy.
    Arm physical state, contacts, course heading, velocity and lateral position are privileged.
    Control runs at 50 Hz with ten 500 Hz servo updates per transition.
    """

    def __init__(self, num_envs, cfg, joints_cfg, show_viewer=False, record_camera=False, root_dir="."):
        self.num_envs = num_envs
        self.device = gs.device
        self.cfg = cfg
        e = cfg["env"]
        self.env_cfg = e
        self.realism_curriculum_cfg = cfg.get("training", {}).get("curriculum", {})
        # Evaluation defaults to full configured realism. A training runner can
        # explicitly choose an earlier curriculum point before resetting.
        self.training_iteration = None
        self.training_realism_alpha = 1.0
        self.training_curriculum_active = False
        self._validate_training_inputs(cfg, joints_cfg)

        self.dt = e["control_dt"]
        self.max_episode_length = math.ceil(e["episode_length_s"] / self.dt)
        self.resample_steps = max(1, int(round(e["resample_commands_s"] / self.dt)))
        self.clip_actions = e["clip_actions"]
        self.reset_noise = e["reset_dof_noise"]
        self.force_thr = e["contact_force_threshold"]
        self.delay_steps = int(e.get("action_delay_steps", 0))
        self.servo_cfg = e.get("servo", {})
        self.servo_safety_cfg = e.get("servo_safety", {})
        self.stall_torque_fraction = float(self.servo_safety_cfg.get("stall_torque_fraction", 0.7))
        self.stall_speed_rad_s = float(self.servo_safety_cfg.get("stall_speed_rad_s", 0.2))
        self.max_stall_duration_s = float(self.servo_safety_cfg.get("max_stall_duration_s", 0.5))
        self.servo_mode = self.servo_cfg.get("mode", "pd")
        if self.servo_mode not in ("pd", "torque_speed"):
            raise ValueError("env.servo.mode must be pd or torque_speed")
        if self.servo_mode == "pd" and cfg["rewards"]["scales"].get("torques", 0) != 0:
            raise ValueError("The torques reward requires torque_speed mode; use zero in PD recovery")
        self.dr_cfg = e.get("domain_randomization", {})
        self.noise_cfg = e.get("observation_noise", {})
        self.sensor_cfg = e.get("joint_sensor", {})
        self.sensor_delay = int(self.sensor_cfg.get("delay_steps", 0))
        if self.sensor_delay < 0:
            raise ValueError("Joint sensor delay must be nonnegative")
        for key in ("position_noise_std_rad", "position_noise_uniform_rad", "quantization_rad"):
            value = float(self.sensor_cfg.get(key, 0.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError("joint_sensor."+key+" must be finite and nonnegative")
        self.push_cfg = e.get("pushes", {})
        self.collect_diagnostics = bool(e.get("collect_diagnostics", False))
        self.heading_feedback = bool(e.get("heading_feedback", False))
        self.critic_lateral_position = bool(e.get("critic_lateral_position", False))
        self.imu_enabled = True
        self.command_reference_frame = e.get("command_reference_frame", "reset_heading")
        self.motor_kernels = None
        requested = cfg.get("performance", {}).get("backend", "auto")
        device = torch.device(self.device)
        self.compute_backend = "warp" if requested != "torch" and device.type == "cuda" else "torch"
        if requested == "warp" and device.type != "cuda":
            raise ValueError("Warp requires CUDA; use --backend auto or torch on CPU")
        self.joint_observation = e.get("joint_observation", "mixed")
        if self.joint_observation != "mixed":
            raise ValueError("joint_observation must be mixed")
        self.physics_steps = int(e["substeps"]) if self.servo_mode == "torque_speed" else 1
        if self.dt <= 0 or self.physics_steps < 1 or self.delay_steps < 0:
            raise ValueError("control_dt/substeps must be positive and delay nonnegative")
        self.physics_dt = self.dt / self.physics_steps
        self.hard_velocity_limit = None
        delay_range = self.dr_cfg.get("action_delay_steps", [self.delay_steps, self.delay_steps])
        self.delay_range = [int(x) for x in delay_range] if self.dr_cfg.get("enabled", False) else [self.delay_steps]*2
        if len(self.delay_range) != 2 or not 0 <= self.delay_range[0] <= self.delay_range[1]:
            raise ValueError("domain_randomization.action_delay_steps must be ordered nonnegative integers")
        self.max_delay = self.delay_range[1]
        self.filter_tau = float(self.servo_cfg.get("target_filter_tau_s", 0.0))
        if self.filter_tau < 0:
            raise ValueError("target_filter_tau_s must be nonnegative")
        self.ground_support_only = bool(e.get("ground_support_only", False))
        self._contact_compat_export = False
        self._contact_legacy_export = False
        self.friction_cfg = e.get("friction")
        cone = e.get("friction_cone", "pyramidal")
        if cone not in ("pyramidal", "elliptic"):
            raise ValueError("friction_cone must be pyramidal or elliptic")
        self.friction_cone = getattr(gs.friction_cone, cone)
        self.max_tilt = e["terminate_tilt_rad"]
        self.min_height = e["terminate_base_height"]
        self.presentation = None
        presentation_cls = None
        presentation_view = None

        self.obs_scales = cfg["obs_scales"]
        self.cmd_scale = torch.tensor(self.obs_scales["commands"], device=self.device, dtype=gs.tc_float)
        self.cmd_ranges = cfg["commands"]
        self.command_curriculum_cfg = self.cmd_ranges.get("curriculum", {})
        self.command_endpoint_probability = self._validate_endpoint_probability(
            self.cmd_ranges.get("lin_vel_x_endpoint_probability",0.0))
        if self.heading_feedback and any(float(v) != 0 for k in ("lin_vel_y", "ang_vel") for v in self.cmd_ranges[k]):
            raise ValueError("heading_feedback requires zero lateral/yaw command ranges")
        self.gait = cfg["gait"]
        self.rew_cfg = cfg["rewards"]
        if float(self.rew_cfg["scales"].get("feet_swing_clearance", 0.0)) != 0.0:
            minimum = float(self.gait.get("minimum_swing_clearance_m", 0.015))
            if not math.isfinite(minimum) or not 0 < minimum <= float(self.gait["swing_height"]):
                raise ValueError("minimum_swing_clearance_m must be positive and no higher than swing_height")
        self.fixed_command = None

        # ---------------------------- joint parameters ---------------------------- #
        d = joints_cfg["defaults"]
        joints = [{**d, **j} for j in joints_cfg["joints"]]
        validate_joint_config({"joints": joints})
        self.joint_names = [j["name"] for j in joints]
        self.num_motors = len(joints)
        self.num_actions = LEG_JOINT_COUNT
        self.policy_joint_names = self.joint_names[:LEG_JOINT_COUNT]
        self.external_arm_joint_names = self.joint_names[LEG_JOINT_COUNT:]

        def vec(key):
            return torch.tensor([float(j[key]) for j in joints], device=self.device, dtype=gs.tc_float)

        self.default_dof_pos = vec("default_pos")
        self.pd_trim = vec("pd_trim")
        self.action_scale = vec("action_scale")
        self.kp = [float(j["kp"]) for j in joints]
        self.kd = [float(j["kd"]) for j in joints]
        self.torque_limit = [float(j["torque_limit"]) if j["torque_limit"] is not None else 1.0e6 for j in joints]
        self.max_speed = torch.tensor(
            [float(j["max_speed"]) if j.get("max_speed") is not None else 1.0e6 for j in joints],
            device=self.device, dtype=gs.tc_float,
        )
        self.nominal_servo_torque = torch.tensor(self.torque_limit, device=self.device, dtype=gs.tc_float)
        self.nominal_servo_speed = self.max_speed.clone()
        nominal_leg = e["nominal_leg_actuator"]
        self.nominal_servo_torque[:LEG_JOINT_COUNT] = float(nominal_leg["torque_limit_nm"])
        self.nominal_servo_speed[:LEG_JOINT_COUNT] = float(nominal_leg["speed_rad_s"])
        self.ankle_roll_idx = [i for i, j in enumerate(joints) if j.get("role") == "ankle_roll"]
        leg_target_slew = float(self.servo_cfg["target_slew_speed_rad_s"])
        arm_target_slew = float(self.servo_cfg["arm_target_slew_speed_rad_s"])
        mandible_target_slew = float(self.servo_cfg["mandible_target_slew_speed_rad_s"])
        leg_hard_limit = float(self.servo_cfg["hard_velocity_limit_rad_s"])
        arm_hard_limit = float(self.servo_cfg["arm_hard_velocity_limit_rad_s"])
        mandible_hard_limit = float(self.servo_cfg["mandible_hard_velocity_limit_rad_s"])
        self.target_slew_speed = torch.tensor(
            [mandible_target_slew if j["role"] == "mandible" else arm_target_slew if j["role"] == "shoulder" else leg_target_slew for j in joints],
            device=self.device, dtype=gs.tc_float,
        )
        self.hard_velocity_limit = torch.tensor(
            [mandible_hard_limit if j["role"] == "mandible" else arm_hard_limit if j["role"] == "shoulder" else leg_hard_limit for j in joints],
            device=self.device, dtype=gs.tc_float,
        )
        self.measured_joint_names = list(self.sensor_cfg.get("measured_joints", ()))
        self.command_estimated_joint_names = list(self.sensor_cfg.get("command_estimated_joints", ()))
        self.measured_joint_indices = [self.joint_names.index(name) for name in self.measured_joint_names]
        self.command_estimated_joint_indices = [self.joint_names.index(name) for name in self.command_estimated_joint_names]
        bounds = joints_cfg.get("target_limit", [-math.pi, math.pi])
        self.target_lo = torch.tensor([float(j.get("target_limit", bounds)[0]) for j in joints], device=self.device, dtype=gs.tc_float)
        self.target_hi = torch.tensor([float(j.get("target_limit", bounds)[1]) for j in joints], device=self.device, dtype=gs.tc_float)

        # ---------------------------------- scene --------------------------------- #
        vis_options = None
        floor_visual = None
        if show_viewer or record_camera:
            # Keep viewer plugins and rendering helpers out of headless training.
            from viewer_scene import PolicyPresentation, floor_surface, options, settings
            viewer_options, vis_options = options(cfg, show_viewer)
            presentation_view = settings(cfg)
            presentation_cls = PolicyPresentation
            floor_visual = floor_surface()
        else:
            viewer_options = gs.options.ViewerOptions(
                camera_pos=(0.6, -0.6, 0.3),
                camera_lookat=(0.0, 0.0, 0.08),
                camera_fov=40,
            )
        scene_options = {}
        if vis_options is not None:
            scene_options["vis_options"] = vis_options
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.physics_dt, substeps=1 if self.servo_mode == "torque_speed" else e["substeps"]),
            viewer_options=viewer_options,
            rigid_options=self._make_rigid_options(e),
            show_viewer=show_viewer,
            **scene_options,
        )
        plane_mat = None
        if self.friction_cfg and self.friction_cfg.get("floor") is not None:
            plane_mat = gs.materials.Rigid(friction=float(self.friction_cfg["floor"]))
        plane = gs.morphs.Plane(tile_size=(0.8, 0.8)) if floor_visual is not None else gs.morphs.Plane()
        floor_options = {"surface": floor_visual} if floor_visual is not None else {}
        self.floor = self.scene.add_entity(plane, material=plane_mat, **floor_options)

        src_xml = os.path.abspath(os.path.join(root_dir, e["xml_path"]))

        self.base_init_pos = torch.tensor(e["init_base_pos"], device=self.device, dtype=gs.tc_float)
        self.base_init_pos[2] += e["spawn_height_margin"]
        self.base_init_quat = torch.tensor(e.get("init_base_quat", [1.0, 0.0, 0.0, 0.0]), device=self.device, dtype=gs.tc_float)
        self._training_model_temp, self.training_xml_path = self._prepare_training_model(src_xml, e)
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file=self.training_xml_path,
                pos=tuple(self.base_init_pos.tolist()),
                quat=tuple(self.base_init_quat.tolist()),
                decimate=False,
            )
        )

        self.cam = None
        if record_camera:
            view = presentation_view
            self.cam = self.scene.add_camera(
                res=tuple(view["record_resolution"]), pos=(0.58, -0.72, 0.38),
                lookat=(0.0, 0.0, 0.125), fov=view["camera_fov"], GUI=False
            )

        self._add_extra_entities()
        self.scene.build(n_envs=num_envs)
        self.nominal_mass = self.robot.get_mass().clone() if "mass_scale" in self.dr_cfg else None
        if torch.any(self.max_speed <= 0) or any(v <= 0 for v in self.torque_limit) or any(v <= 0 for v in self.kp) or any(v < 0 for v in self.kd):
            raise ValueError("Servo speed, torque and kp must be positive; kd nonnegative")
        if not 0 < self.gait["stance_ratio"] < 1 or self.gait["period_s"] <= 0 or self.gait["swing_height"] < 0:
            raise ValueError("Gait period must be positive, stance_ratio in (0,1), height nonnegative")

        # ---------------------------- indices / actuators -------------------------- #
        print("Links:", [l.name for l in self.robot.links])
        self.motor_dofs = [int(list(self.robot.get_joint(n).dofs_idx_local)[0]) for n in self.joint_names]
        self.motor_qs = [int(list(self.robot.get_joint(n).qs_idx_local)[0]) for n in self.joint_names]
        self.robot.set_dofs_kp(self.kp, self.motor_dofs)
        self.robot.set_dofs_kv(self.kd, self.motor_dofs)
        self.robot.set_dofs_force_range(
            (-self.nominal_servo_torque).tolist(), self.nominal_servo_torque.tolist(), self.motor_dofs
        )

        self._apply_foot_friction()
        self.push_dofs = None
        if self.push_cfg.get("enabled", False):
            free_joint = self.robot.get_joint("floating_base")
            free_dofs = list(free_joint.dofs_idx_local)
            if len(free_dofs) != 6:
                raise ValueError("Pushes require the six-DOF floating_base joint")
            self.push_dofs = [int(i) for i in free_dofs[:2]]
            if float(self.push_cfg.get("interval_s", 5.0)) <= 0:
                raise ValueError("pushes.interval_s must be positive")

        self.base_idx = self._link_idx(e["base_link"])
        self.feet_idx = [self._link_idx(n) for n in e["foot_links"]]
        self.term_idx = [self._link_idx(n) for n in e["termination_links"]]
        self.pen_idx = [self._link_idx(n) for n in e["penalized_links"]]
        self.nonfoot_idx = [i for i in range(len(self.robot.links)) if i not in self.feet_idx]
        # Dynamic arm/leg clusters exclude the chassis so mounting contacts are
        # never mistaken for an arm obstruction.
        arm_tokens = ("MG90S", "Suporte-Servo", "chas_third_link", "chas_second_link")
        leg_tokens = ("hipforward", "thigh-v2", "knee-v3", "servo.MG", "joiner2", "mg996_arm", "linky-v2", "loong_rightfootpad")
        self.arm_link_flags = torch.tensor(
            [any(token in link.name for token in arm_tokens) for link in self.robot.links],
            device=self.device, dtype=torch.bool)
        self.leg_link_flags = torch.tensor(
            [any(token in link.name for token in leg_tokens) for link in self.robot.links],
            device=self.device, dtype=torch.bool)
        self.arm_leg_obstruction_force_n = float(e.get("external_arms", {}).get("obstruction_force_n", 1.0))
        self.arm_leg_obstruction_duration_s = float(e.get("external_arms", {}).get("obstruction_duration_s", 0.10))

        # ---------------------------------- buffers -------------------------------- #
        n, na, dev, f = num_envs, self.num_motors, self.device, gs.tc_float
        self.arm_leg_obstruction_time = torch.zeros((n,), device=dev, dtype=f)
        self.arm_leg_contact = torch.zeros((n,), device=dev, dtype=torch.bool)
        self.arm_leg_contact_force = torch.zeros((n,), device=dev, dtype=f)
        self.episode_realism_alpha = torch.ones((n,), device=dev, dtype=f)
        self.episode_nominal_replay = torch.zeros((n,), device=dev, dtype=torch.bool)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=dev, dtype=f).repeat(n, 1)
        self.frame_obs_buf = torch.zeros((n, 46), device=dev, dtype=f)
        self.obs_history = torch.zeros((n, 5, 46), device=dev, dtype=f)
        self.obs_buf = torch.zeros((n, 230), device=dev, dtype=f)
        self.critic_buf = torch.zeros((n, 53), device=dev, dtype=f)
        self.obs_position_buf = torch.empty((n, na), device=dev, dtype=f)
        self.rew_buf = torch.zeros((n,), device=dev, dtype=f)
        self.reset_buf = torch.ones((n,), device=dev, dtype=torch.bool)
        self.time_out_buf = torch.zeros((n,), device=dev, dtype=torch.bool)
        self.episode_length_buf = torch.zeros((n,), device=dev, dtype=torch.long)
        self.commands = torch.zeros((n, 3), device=dev, dtype=f)
        self.heading_reference = torch.zeros((n,), device=dev, dtype=f)
        self.lateral_origin = self.base_init_pos[1].repeat(n)
        self.actions = torch.zeros((n, self.num_actions), device=dev, dtype=f)
        self.action_delay = torch.full((n,), self.delay_steps, device=dev, dtype=torch.long)
        self.servo_kp = torch.tensor(self.kp, device=dev, dtype=f).repeat(n, 1)
        self.servo_kd = torch.tensor(self.kd, device=dev, dtype=f).repeat(n, 1)
        self.servo_torque = self.nominal_servo_torque.repeat(n, 1)
        self.servo_speed = self.nominal_servo_speed.repeat(n, 1)
        self.hard_speed_cap = self.hard_velocity_limit.unsqueeze(0).repeat(n, 1)
        self.servo_target_slew = self.target_slew_speed.unsqueeze(0).repeat(n, 1)
        self.servo_deadband = torch.full((n, na), float(self.servo_cfg.get("deadband_rad", 0.0)), device=dev, dtype=f)
        # Total flank-to-flank play, independent of the electrical position deadband.
        self.servo_backlash = torch.full((n, na), float(self.servo_cfg.get("backlash_rad", 0.0)), device=dev, dtype=f)
        self.slew_sat = torch.zeros((n, na), device=dev, dtype=torch.bool)
        self.torque_sat = torch.zeros_like(self.slew_sat)
        self.speed_sat = torch.zeros_like(self.slew_sat)
        self.applied_torque = torch.zeros((n, na), device=dev, dtype=f)
        self.joint_torque_square_sum = torch.zeros((na,), device=dev, dtype=f)
        self.joint_torque_peak_abs = torch.zeros((na,), device=dev, dtype=f)
        self.joint_torque_clip_count = torch.zeros((na,), device=dev, dtype=f)
        self.joint_speed_clip_count = torch.zeros((na,), device=dev, dtype=f)
        self.joint_torque_sample_count = 0
        self.motor_stall_duration = torch.zeros((n, na), device=dev, dtype=f)
        self.motor_stall_failure = torch.zeros((n,), device=dev, dtype=torch.bool)
        self.joint_stall_duration_peak = torch.zeros((na,), device=dev, dtype=f)
        self.joint_stall_event_count = torch.zeros((na,), device=dev, dtype=f)
        self.torque_effort = torch.zeros((n,), device=dev, dtype=f)
        self.servo_rest = torch.clamp(self.default_dof_pos + self.pd_trim, self.target_lo, self.target_hi)
        self.servo_target = self.servo_rest.unsqueeze(0).repeat(n, 1)
        self.servo_play_target = self.servo_target.clone()
        self.external_arms = ExternalArmCommands(n, cfg, dev, f, self.dt,
                                                self.target_lo[LEG_JOINT_COUNT:], self.target_hi[LEG_JOINT_COUNT:])
        # Histories contain absolute twelve-motor targets. The actor supplies
        # only the first eight; upper-body columns come exclusively from external_arms.
        self.motor_target_hist = self.servo_rest.repeat(n, self.max_delay + 1, 1)
        # Software observer uses only nominal issued commands, not hidden physics.
        self.observer_hist = self.servo_rest.repeat(n, 2, 1)
        self.observer_pos = self.servo_target.clone()
        self.observer_vel = torch.zeros_like(self.observer_pos)
        measured_defaults = self.default_dof_pos[self.measured_joint_indices]
        self.sensor_hist = measured_defaults.repeat(n, self.sensor_delay+1, 1)
        self.sensor_pos = measured_defaults.repeat(n, 1)
        self.previous_dof_vel = torch.zeros_like(self.observer_pos)
        self.dof_acc = torch.zeros_like(self.observer_pos)
        self.last_actions = torch.zeros_like(self.actions)
        self.phase = torch.zeros((n,), device=dev, dtype=f)
        self.phase_offset = (torch.arange(n, device=dev) % 2).to(f) * 0.5
        self.reset_count = torch.zeros(n, device=dev, dtype=torch.long)
        self.stance_mask = torch.ones((n, 2), device=dev, dtype=torch.bool)
        self.foot_contact = torch.zeros((n, 2), device=dev, dtype=torch.bool)
        self.extras = {"observations": {}}
        self.training_velocity_sums = torch.zeros((3,), device=dev, dtype=f)
        self.training_velocity_steps = 0
        self.training_episode_count = torch.zeros((), device=dev, dtype=f)
        self.training_episode_duration_sum_s = torch.zeros((), device=dev, dtype=f)
        self.training_fall_count = torch.zeros((), device=dev, dtype=f)
        self.training_stall_count = torch.zeros((), device=dev, dtype=f)
        self.training_arm_obstruction_count = torch.zeros((), device=dev, dtype=f)
        self.training_swing_clearance_sum = torch.zeros((2,), device=dev, dtype=f)
        self.training_swing_samples = torch.zeros((2,), device=dev, dtype=f)
        self.training_swing_peaks = torch.zeros((n, 2), device=dev, dtype=f)
        self.training_swing_seen = torch.zeros((n, 2), device=dev, dtype=torch.bool)
        self.training_servo_sums = torch.zeros((4,), device=dev, dtype=f)
        self.hard_speed_limited = torch.zeros((n, na), device=dev, dtype=torch.bool)
        self.course_metric_sums = torch.zeros(2, device=dev, dtype=f)
        self.completed_peak_sums = torch.zeros(3, device=dev, dtype=f)
        self.completed_pair_count = torch.zeros((), device=dev, dtype=f)
        self.completed_valid_pair_count = torch.zeros((), device=dev, dtype=f)
        self.completed_pair_balance_sum = torch.zeros((), device=dev, dtype=f)
        self.completed_time_gap_sum = torch.zeros((), device=dev, dtype=f)
        self.completed_time_relative_gap_sum = torch.zeros((), device=dev, dtype=f)
        self.clearance_pair_enabled = any(float(self.rew_cfg["scales"].get(k, 0)) != 0
                                          for k in ("feet_swing_symmetry", "feet_swing_peak"))
        if self.clearance_pair_enabled and float(self.gait["swing_height"]) <= 0:
            raise ValueError("Paired swing rewards require a positive swing_height")
        self.previous_swing = torch.zeros((n, 2), device=dev, dtype=torch.bool)
        self.swing_started = torch.zeros_like(self.previous_swing)
        self.swing_airborne_time = torch.zeros((n, 2), device=dev, dtype=f)
        self.swing_running_peak = torch.zeros((n, 2), device=dev, dtype=f)
        self.completed_swing_peaks = torch.zeros_like(self.swing_running_peak)
        self.swing_pair_ready = torch.zeros_like(self.previous_swing)
        self.completed_swing_pair = torch.zeros(n, device=dev, dtype=torch.bool)
        self.completed_swing_pair_valid = torch.zeros(n, device=dev, dtype=torch.bool)
        self.completed_swing_balance = torch.zeros(n, device=dev, dtype=f)
        self.completed_swing_times = torch.zeros((n, 2), device=dev, dtype=f)
        self.completed_swing_time_gap = torch.zeros(n, device=dev, dtype=f)
        self.completed_swing_time_relative_gap = torch.zeros(n, device=dev, dtype=f)
        self.completed_swing_peak_gap = torch.zeros(n, device=dev, dtype=f)
        self.imu = SixAxisIMU(n, e["imu"], self.dt, dev, f)

        # A fall is a one-off cost, independent of control rate.
        event_terminal = bool(self.rew_cfg.get("termination_is_event", False))
        self.reward_scales = {
            k: float(v) * (1.0 if k == "termination" and event_terminal else self.dt)
            for k, v in self.rew_cfg["scales"].items()
        }
        self.episode_reward_names = tuple(self.reward_scales)
        self.episode_sum_matrix = torch.zeros((len(self.episode_reward_names), n), device=dev, dtype=f)
        self.episode_sums = {
            name: self.episode_sum_matrix[i]
            for i, name in enumerate(self.episode_reward_names)
        }
        self._reward_fns = {k: getattr(self, "_reward_" + k) for k in self.reward_scales}

        # Cache collision-pad vertices in the authored foot frame. Clearance is
        # always above the actual plane z=0, independent of spawn/drop margin.
        self.sole_vertices = None
        self.foot_clearance = torch.zeros((n, 2), device=dev, dtype=f)
        self.foot_flatness = torch.zeros((n, 2), device=dev, dtype=f)
        self.reset_idx(torch.arange(n, device=dev), noise=False)
        self._update_state()
        self._cache_sole_vertices()
        self._compute_sole_clearance()
        self.reset()
        if self.compute_backend == "warp":
            try:
                from cuda_kernels import MotorKernels
                self.motor_kernels = MotorKernels(self)
            except ImportError as error:
                raise RuntimeError("CUDA auto/warp requires warp-lang; install requirements-training.txt or select --backend torch") from error
        print(f"[compute] backend={self.compute_backend}; leg_actions=8 external_arms=4 actor=230 (history=5x46) critic=53; no contact switches")
        if presentation_cls is not None:
            self.presentation = presentation_cls(self, show_viewer)

    def _validate_training_inputs(self, cfg, joints_cfg):
        validate_joint_config(joints_cfg)
        validate_realistic_servo_limits(cfg, joints_cfg)
        validate_training_contract(cfg, joints_cfg)
        validate_curriculum_configuration(cfg)

    def _make_rigid_options(self, env_cfg):
        """Default walking solver; isolated tasks may set explicit solver options."""
        return gs.options.RigidOptions(
            enable_collision=True,
            enable_self_collision=env_cfg["self_collision"],
            enable_adjacent_collision=env_cfg["adjacent_collision"],
            enable_neutral_collision=env_cfg["neutral_collision"],
            max_collision_pairs=int(env_cfg.get("max_collision_pairs", 256)),
            max_contacts=int(env_cfg.get("max_contacts", 256)),
            enable_joint_limit=True,
            friction_cone=self.friction_cone,
            batch_dofs_info=True,
            batch_links_info=bool(env_cfg.get("batch_links_info", False) or
                (self.dr_cfg.get("enabled", False) and "mass_scale" in self.dr_cfg)),
        )

    def _prepare_training_model(self, source, env_cfg):
        """Extension point for isolated fine-tuning tasks with additional hardware."""
        return prepare_training_mjcf(source, env_cfg)

    def _add_extra_entities(self):
        """Extension point before build; ordinary walking adds no other entities."""
        return None

    # ------------------------------------------------------------------------- #
    def _link_idx(self, name):
        try:
            return int(self.robot.get_link(name).idx_local)
        except Exception as ex:
            raise RuntimeError(
                f"Link '{name}' not found in the Genesis entity ({ex}).\n"
                f"Available links: {[l.name for l in self.robot.links]}\n"
                "Fix the name in config.yaml."
            )

    # ------------------------------------------------------------------------- #
    def _apply_foot_friction(self):
        fr = self.friction_cfg
        if not fr or fr.get("foot") is None:
            return
        val = float(fr["foot"])
        geoms = [g for n in self.env_cfg["foot_links"] for g in self.robot.get_link(n).geoms]
        try:
            for g in geoms:
                g.set_friction(val)
        except Exception as ex:
            print(f"[warn] could not set foot friction ({ex}); XML foot friction stays in effect")
            return
        try:
            print(f"[friction] floor={fr.get('floor')} foot geoms={[round(float(g.get_friction()), 3) for g in geoms]}")
        except Exception:
            print(f"[friction] requested floor={fr.get('floor')} foot={val} (could not read back)")

    def set_training_iteration(self, iteration):
        """Update reward weights and select realism for newly reset episodes."""
        iteration = int(iteration)
        if iteration < 0:
            raise ValueError("training iteration must be nonnegative")
        self.training_iteration = iteration
        penalty_schedule = self.cfg.get("training", {}).get("reward_penalty_curriculum")
        if penalty_schedule is None:
            self.reward_penalty_alpha = 1.0
        else:
            warmup = int(penalty_schedule["warmup_iterations"])
            ramp = int(penalty_schedule["ramp_iterations"])
            if warmup < 0 or ramp < 1:
                raise ValueError("reward penalty curriculum requires warmup >= 0 and ramp >= 1")
            # Iteration is zero-based: ramp after warmup across the configured updates.
            self.reward_penalty_alpha = min(1.0, max(0.0, (iteration - warmup + 1) / ramp))
        self.external_arms.set_training_iteration(iteration)
        if not self.realism_curriculum_cfg.get("enabled", False):
            return self.get_training_curriculum_state()
        self.training_curriculum_active = True
        warmup = int(self.realism_curriculum_cfg.get("nominal_iterations", 100))
        ramp = int(self.realism_curriculum_cfg.get("ramp_iterations", 1500))
        if warmup < 0 or ramp < 1:
            raise ValueError("realism curriculum requires nominal_iterations >= 0 and ramp_iterations >= 1")
        self.training_realism_alpha = min(1.0, max(0.0, (iteration - warmup) / ramp))
        return self.get_training_curriculum_state()

    def get_training_curriculum_state(self):
        """Return the curriculum point for checkpoint metadata."""
        return {
            "iteration": self.training_iteration,
            "realism_alpha": self.training_realism_alpha,
            "reward_penalty_alpha": getattr(self, "reward_penalty_alpha", 1.0),
            "arm_pose_alpha": self.external_arms.pose_alpha,
            "nominal_replay_probability": float(self.realism_curriculum_cfg.get("nominal_replay_probability", 0.0)),
            "tracking_sigma_lin": float(self.rew_cfg["tracking_sigma_lin"]),
            "reward_version": REWARD_VERSION,
        }

    def load_training_curriculum_state(self, state):
        """Restore the curriculum point from runner checkpoint metadata."""
        if not isinstance(state, dict) or state.get("iteration") is None:
            raise ValueError("curriculum state must contain a nonnegative iteration")
        return self.set_training_iteration(state["iteration"])

    def _interpolated_random_range(self, key, alpha, default=(1.0, 1.0)):
        full_lo, full_hi = self.dr_cfg.get(key, default)
        full_lo, full_hi = float(full_lo), float(full_hi)
        return 1.0 + alpha * (full_lo - 1.0), 1.0 + alpha * (full_hi - 1.0)

    # ------------------------------------------------------------------------- #
    def _randomize_actuators(self, envs_idx):
        """Draw independent actuator properties once per episode, never mid-step."""
        if not self.dr_cfg.get("enabled", False):
            # The nominal buffers and engine parameters were initialized after
            # scene.build(); a non-randomized reset must not resample or reinstall them.
            return
        n = len(envs_idx)
        alpha = self.episode_realism_alpha[envs_idx].unsqueeze(1)
        for name, base in (("kp", self.kp), ("kd", self.kd),
                           ("torque", self.torque_limit), ("speed", self.max_speed)):
            lo, hi = self._interpolated_random_range(name + "_scale", alpha)
            base = torch.as_tensor(base, device=self.device, dtype=gs.tc_float)
            getattr(self, "servo_" + name)[envs_idx] = base * rand_uniform(lo, hi, (n, self.num_motors), self.device)
        value = float(self.servo_cfg.get("deadband_rad", 0.0))
        full_lo, full_hi = self.dr_cfg.get("deadband_rad", [value, value])
        lo = value + alpha * (float(full_lo) - value)
        hi = value + alpha * (float(full_hi) - value)
        self.servo_deadband[envs_idx] = rand_uniform(lo, hi, (n, self.num_motors), self.device)
        width = float(self.servo_cfg.get("backlash_rad", 0.0))
        full_lo, full_hi = self.dr_cfg.get("backlash_rad", [width, width])
        lo = width + alpha * (float(full_lo) - width)
        hi = width + alpha * (float(full_hi) - width)
        self.servo_backlash[envs_idx] = rand_uniform(lo, hi, (n, self.num_motors), self.device)
        sampled_delay = torch.randint(self.delay_range[0], self.delay_range[1]+1, (n,), device=self.device)
        self.action_delay[envs_idx] = torch.round(
            self.delay_steps + (sampled_delay - self.delay_steps) * alpha.squeeze(1)).long()
        if "friction_scale" in self.dr_cfg:
            lo, hi = self._interpolated_random_range("friction_scale", alpha)
            ratio = rand_uniform(lo, hi, (n, 1), self.device).repeat(1, len(self.feet_idx))
            self.robot.set_friction_ratio(ratio, self.feet_idx, envs_idx=envs_idx)
        if self.nominal_mass is not None:
            lo, hi = self._interpolated_random_range("mass_scale", alpha)
            nominal = self.nominal_mass if self.nominal_mass.ndim == 0 else self.nominal_mass[envs_idx]
            self.robot.set_mass(nominal * rand_uniform(lo.squeeze(1), hi.squeeze(1), (n,), self.device), envs_idx=envs_idx)
        self.robot.set_dofs_kp(self.servo_kp[envs_idx], self.motor_dofs, envs_idx=envs_idx)
        self.robot.set_dofs_kv(self.servo_kd[envs_idx], self.motor_dofs, envs_idx=envs_idx)
        self.robot.set_dofs_force_range(-self.servo_torque[envs_idx], self.servo_torque[envs_idx], self.motor_dofs, envs_idx=envs_idx)

    def _servo_force(self, q, qd):
        # Stateful play operator on the servo transmission target. A reversal
        # must traverse the full configured gap before the opposite flank moves.
        # This approximates lost motion without adding gear/motor inertia DOFs.
        half_width = 0.5 * self.servo_backlash
        self.servo_play_target.copy_(torch.maximum(
            torch.minimum(self.servo_play_target, self.servo_target + half_width),
            self.servo_target - half_width))
        error = self.servo_play_target - q
        error = error.sign() * (error.abs() - self.servo_deadband).clamp_min(0.0)
        requested = self.servo_kp * error - self.servo_kd * qd
        # Linear DC motor envelope in the driving direction, full torque to brake.
        # This is a no-load motor curve, not a shaft-speed clamp: external loads
        # can back-drive the joint above no-load speed while torque is bounded.
        driving = requested * qd >= 0
        fraction = (1.0 - qd.abs() / self.servo_speed.clamp_min(1e-6)).clamp(0, 1)
        limit = self.servo_torque * torch.where(driving, fraction, torch.ones_like(fraction))
        force = torch.maximum(torch.minimum(requested, limit), -limit)
        clipped = requested.abs() > limit
        return force, clipped, clipped & driving & (qd.abs() > 0)

    def _update_motor_stall(self):
        """Track consecutive high-torque, near-zero-speed joints at control rate."""
        previous_duration = self.motor_stall_duration.clone()
        normalized_torque = self.applied_torque.abs() / self.servo_torque.clamp_min(1e-6)
        stalled = ((normalized_torque >= self.stall_torque_fraction)
                   & (self.dof_vel.abs() <= self.stall_speed_rad_s))
        self.motor_stall_duration = torch.where(
            stalled, self.motor_stall_duration + self.dt, torch.zeros_like(self.motor_stall_duration))
        crossed = (self.motor_stall_duration >= self.max_stall_duration_s) & (previous_duration < self.max_stall_duration_s)
        self.joint_stall_event_count += crossed.to(gs.tc_float).sum(dim=0)
        self.joint_stall_duration_peak.copy_(torch.maximum(
            self.joint_stall_duration_peak, self.motor_stall_duration.amax(dim=0)))
        # Exogenous upper-body load must not terminate a leg policy's rollout.
        self.motor_stall_failure = (self.motor_stall_duration[:, :LEG_JOINT_COUNT] >= self.max_stall_duration_s).any(dim=1)

    def _sample_joint_positions(self, positions, envs_idx=None):
        """Apply sensor error to a fresh read, including the initial reset read.

        position_noise_uniform_rad is a symmetric half-width, not a Gaussian
        standard deviation. Bootstrap disables potentiometer noise and quantization; only the
        eight measured leg joints enter this sensor path.
        """
        sample = positions.clone()
        if positions.shape[-1] != len(self.measured_joint_indices):
            raise ValueError("Only configured leg potentiometers may enter the measured sensor path")
        alpha_source = self.episode_realism_alpha if envs_idx is None else self.episode_realism_alpha[envs_idx]
        alpha = alpha_source.unsqueeze(1)
        std = float(self.sensor_cfg.get("position_noise_std_rad", 0.0))
        if std > 0:
            sample += torch.randn_like(sample)*std*alpha
        configured_half_width = float(self.sensor_cfg.get("position_noise_uniform_rad", 0.0))
        if configured_half_width > 0:
            half_width = configured_half_width*alpha
            sample += (2.0*torch.rand_like(sample)-1.0)*half_width
        configured_quantum = float(self.sensor_cfg.get("quantization_rad", 0.0))
        if configured_quantum > 0:
            quantum = configured_quantum*alpha
            quantized = torch.round(sample/quantum.clamp_min(1e-12))*quantum
            sample = torch.where(alpha > 0, quantized, sample)
        return sample

    def _update_joint_sensor(self):
        if not self.sensor_cfg:
            return
        self.sensor_hist = torch.cat([self.sensor_hist[:, 1:], self.dof_pos[:, self.measured_joint_indices].unsqueeze(1)], dim=1)
        # Delay raw positions first, then acquire one noisy reading at 50 Hz.
        self.sensor_pos = self._sample_joint_positions(self.sensor_hist[:, 0])
        # No differences of position samples are used for velocity observations.

    def _apply_pushes(self):
        if self.push_dofs is None:
            return
        interval = max(1, int(round(float(self.push_cfg.get("interval_s", 5.0)) / self.dt)))
        selected = ((self.episode_length_buf > 0) & (self.episode_length_buf % interval == 0)).nonzero(as_tuple=False).flatten()
        if len(selected):
            delta = float(self.push_cfg.get("velocity_delta_m_s", 0.05))
            if delta < 0:
                raise ValueError("push velocity delta must be nonnegative")
            vel = self.robot.get_dofs_velocity(self.push_dofs)[selected]
            alpha = self.episode_realism_alpha[selected].unsqueeze(1)
            vel += rand_uniform(-delta, delta, vel.shape, self.device)*alpha
            # Small instantaneous planar velocity impulse on the floating base.
            self.robot.set_dofs_velocity(vel, self.push_dofs, envs_idx=selected)

    def set_fixed_command(self, vx, vy=0.0, yaw=0.0):
        """Override command sampling (used by eval.py)."""
        if (not all(math.isfinite(float(v)) for v in (vx, vy, yaw)) or vy != 0.0
                or not -0.10 <= vx <= 0.15 or not -0.4 <= yaw <= 0.4):
            raise ValueError("Commands require vx in [-0.10, 0.15], yaw in [-0.4, 0.4], and vy=0")
        self.fixed_command = torch.tensor([vx, vy, yaw], device=self.device, dtype=gs.tc_float)
        self.commands[:] = self.fixed_command

    @staticmethod
    def _validate_endpoint_probability(value):
        try:
            probability = float(value)
        except (TypeError,ValueError) as error:
            raise ValueError("lin_vel_x_endpoint_probability must be finite and in [0,1]") from error
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("lin_vel_x_endpoint_probability must be finite and in [0,1]")
        return probability

    def _resample_commands(self, envs_idx):
        """Expand from walking to stops, turns, reverse, and blended joystick commands."""
        if len(envs_idx) == 0:
            return
        if self.fixed_command is not None:
            self.commands[envs_idx] = self.fixed_command
            return
        n = len(envs_idx)
        iteration = self.training_iteration or 0
        stop_alpha = min(1.0, max(0.0, iteration / 1000.0))
        turn_alpha = min(1.0, max(0.0, (iteration - 1000) / 1500.0))
        reverse_alpha = min(1.0, max(0.0, (iteration - 2500) / 1500.0))
        # Masked mixtures retain many familiar forward examples throughout.
        mode = torch.rand(n, device=self.device)
        forward = rand_uniform(0.0, 0.15, (n,), self.device)
        forward = torch.where(torch.rand(n, device=self.device) < 0.35, 0.15, forward)
        reverse = -rand_uniform(0.03, 0.10, (n,), self.device) * reverse_alpha
        sign = torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        yaw = sign * rand_uniform(0.10, 0.40, (n,), self.device) * turn_alpha
        vx = forward.clone()
        wz = torch.zeros_like(vx)
        stop = mode < 0.15 * stop_alpha
        turning = (mode >= 0.15) & (mode < 0.35)
        backing = (mode >= 0.35) & (mode < 0.50)
        mixed = (mode >= 0.50) & (mode < 0.70)
        vx = torch.where(stop, 0.0, vx)
        vx = torch.where(turning & (turn_alpha > 0.0), 0.0, vx)
        vx = torch.where(backing & (reverse_alpha > 0.0), reverse, vx)
        wz = torch.where(turning | mixed, yaw, wz)
        # Mixed commands cover variable forward, reverse and turning together.
        mixed_reverse = mixed & (reverse_alpha > 0.0) & (torch.rand(n, device=self.device) < 0.30)
        vx = torch.where(mixed_reverse, reverse, vx)
        self.commands[envs_idx, 0] = vx
        self.commands[envs_idx, 1] = 0.0
        self.commands[envs_idx, 2] = wz

    # ------------------------------------------------------------------------- #
    def _update_state(self):
        self.base_pos = self.robot.get_pos()
        self.raw_base_quat = self.robot.get_quat()
        # The chassis is authored at yaw pi while the logical IMU site carries
        # the corresponding pi rotation. Compose that fixed frame offset here;
        # the MJCF geometry and raw link transforms remain unchanged.
        rw, rx, ry, rz = self.raw_base_quat.unbind(dim=1)
        self.base_quat = torch.stack((-rz, ry, -rx, rw), dim=1)
        w, x, y, z = self.base_quat.unbind(dim=1)
        self.raw_yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        delta = self.raw_yaw-self.heading_reference
        self.heading_error = torch.atan2(torch.sin(delta), torch.cos(delta))
        self.world_lin_vel = self.robot.get_vel()
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.world_lin_vel)
        c, s = self.heading_reference.cos(), self.heading_reference.sin()
        self.reference_lin_vel = self.world_lin_vel.clone()
        self.reference_lin_vel[:, 0] = c * self.world_lin_vel[:, 0] + s * self.world_lin_vel[:, 1]
        self.reference_lin_vel[:, 1] = -s * self.world_lin_vel[:, 0] + c * self.world_lin_vel[:, 1]
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.robot.get_ang())
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.global_gravity)
        self.dof_pos = self.robot.get_dofs_position(self.motor_dofs)
        self.dof_vel = self.robot.get_dofs_velocity(self.motor_dofs)
        self.link_pos = self.robot.get_links_pos()
        self.foot_quat = self.robot.get_links_quat(self.feet_idx)
        self.link_vel = self.robot.get_links_vel()
        if self.sole_vertices is not None:
            self._compute_sole_clearance()
        self.contact_forces = self.robot.get_links_net_contact_force()
        self.force_norm = self.contact_forces.norm(dim=-1)
        if self.ground_support_only:
            contacts = self._get_floor_contacts()
            self.ground_contact_force = self._sum_ground_normal_forces(contacts)
            self.foot_ground_force = self.ground_contact_force[:, self.feet_idx]
            self.foot_contact = self.ground_contact_force[:, self.feet_idx] > self.force_thr
        else:
            self.foot_contact = self.force_norm[:, self.feet_idx] > self.force_thr

    def _get_floor_contacts(self):
        if not self._contact_compat_export:
            try:
                return self.robot.get_contacts(with_entity=self.floor, is_padded=True)
            except RuntimeError as ex:
                if "Expected dtype int64 for index" not in str(ex):
                    raise
                self._contact_compat_export = True
                print("[contacts] using scoped int64 gather adapter for Genesis/torch compatibility")
        if not getattr(self, "_contact_legacy_export", False):
            try:
                with _ContactGatherIndexCompat():
                    return self.robot.get_contacts(with_entity=self.floor, is_padded=True)
            except RuntimeError as ex:
                if "Expected dtype int64 for index" not in str(ex):
                    raise
                # Last resort for a version whose gather is not mode-dispatchable.
                self._contact_legacy_export = True
                print("[contacts] using legacy copied contact exporter")
        previous = gs.use_zerocopy
        try:
            gs.use_zerocopy = False
            return self.robot.get_contacts(with_entity=self.floor, is_padded=True)
        finally:
            gs.use_zerocopy = previous

    def _sum_ground_normal_forces(self, contacts):
        # get_contacts(with_entity=floor) supplies only robot/floor valid pairs;
        # padded unused/self-contact slots are explicitly masked before scatter.
        n_links = len(self.robot.links)
        start = self.robot.link_start
        forces = torch.zeros((self.num_envs, n_links), device=self.device, dtype=gs.tc_float)
        for side in ("a", "b"):
            link = contacts["link_" + side] - start
            valid = contacts["valid_mask"] & (link >= 0) & (link < n_links)
            normal = contacts["force_" + side][..., 2].clamp_min(0.0)
            forces.scatter_add_(1, link.clamp(0, n_links-1).long(), normal * valid.to(gs.tc_float))
        return forces

    def _nonfoot_ground_contact(self):
        return (self.ground_contact_force[:, self.nonfoot_idx] > self.force_thr).any(dim=1)

    def _update_arm_leg_contacts(self):
        """Preserve physical self contacts and stop sustained arm-leg obstruction."""
        try:
            with _ContactGatherIndexCompat():
                contacts = self.robot.get_contacts(with_entity=self.robot, is_padded=True)
        except (AttributeError, RuntimeError) as error:
            raise RuntimeError("Could not inspect physical arm-leg self contacts; obstruction guard cannot run") from error
        start = self.robot.link_start
        a = contacts["link_a"] - start
        b = contacts["link_b"] - start
        if contacts["valid_mask"].ndim != 2 or contacts["valid_mask"].shape[1] == 0:
            self.arm_leg_contact_force.zero_()
            self.arm_leg_contact.zero_()
            self.arm_leg_obstruction_time.zero_()
            return
        valid = contacts["valid_mask"] & (a >= 0) & (b >= 0) & (a < len(self.robot.links)) & (b < len(self.robot.links))
        a_safe, b_safe = a.clamp(0, len(self.robot.links)-1).long(), b.clamp(0, len(self.robot.links)-1).long()
        a_arm, b_arm = self.arm_link_flags[a_safe], self.arm_link_flags[b_safe]
        a_leg, b_leg = self.leg_link_flags[a_safe], self.leg_link_flags[b_safe]
        pair = valid & ((a_arm & b_leg) | (b_arm & a_leg))
        magnitude = contacts["force_a"].norm(dim=-1)
        strength = torch.where(pair, magnitude, torch.zeros_like(magnitude)).amax(dim=1)
        self.arm_leg_contact_force.copy_(strength)
        self.arm_leg_contact.copy_(strength >= self.arm_leg_obstruction_force_n)
        self.arm_leg_obstruction_time = torch.where(
            self.arm_leg_contact,
            self.arm_leg_obstruction_time + self.dt,
            torch.zeros_like(self.arm_leg_obstruction_time))

    def _cache_sole_vertices(self):
        names = self.env_cfg.get("toe_collision_geoms", ["left_sole_collision", "right_sole_collision"])
        if len(names) != 2:
            raise ValueError("toe_collision_geoms must name the left and right sole collision geoms")
        self.sole_vertices = []
        self.sole_normal_axis = []
        for foot, (link_name, geom_name) in enumerate(zip(self.env_cfg["foot_links"], names)):
            link_geoms = list(self.robot.get_link(link_name).geoms)
            matches = [g for g in link_geoms if g.metadata.get("name") == geom_name]
            if not matches:
                sole_size = torch.tensor([0.065, 0.005, 0.112], device=self.device, dtype=gs.tc_float).sort().values
                origin = self.link_pos[0, self.feet_idx[foot]]
                for geom in link_geoms:
                    vertices = geom.get_verts()
                    vertices = vertices[0] if vertices.ndim == 3 else vertices
                    if vertices.ndim != 2 or len(vertices) == 0:
                        continue
                    quat = self.foot_quat[0, foot].expand(len(vertices), -1)
                    local = quat_rotate_inverse(quat, vertices-origin)
                    extent = (local.amax(0)-local.amin(0)).sort().values
                    if torch.allclose(extent, sole_size, atol=1e-5, rtol=1e-3):
                        matches.append(geom)
            if not matches and len(link_geoms) == 1:
                # Genesis may not keep MJCF geom names; the unambiguous single-geom case remains supported.
                print(f"[sole] '{geom_name}' not found by name; using the only collision geom on {link_name}")
                matches = link_geoms
            if len(matches) != 1:
                raise ValueError("Expected one sole collision geom named " + geom_name + " on " + link_name
                                 + "; geoms found: " + str([g.metadata.get("name") for g in link_geoms]))
            world = matches[0].get_verts()
            world = world[0] if world.ndim == 3 else world
            if world.ndim != 2 or len(world) == 0:
                raise ValueError("Sole collision geom has no usable vertices: " + geom_name)
            origin = self.link_pos[0, self.feet_idx[foot]]
            quat = self.foot_quat[0, foot].expand(len(world), -1)
            local = quat_rotate_inverse(quat, world - origin).detach().clone()
            self.sole_vertices.append(local)
            # The sole is a thin box: its shortest extent is the sole normal (used by feet_flat).
            self.sole_normal_axis.append(int((local.amax(dim=0) - local.amin(dim=0)).argmin()))

    def _compute_sole_clearance(self):
        # Rotate world-up into each local foot frame. A vertex's world z is its
        # dot product with this direction plus the foot origin's world height.
        # Minimum over the complete sole mesh also catches an edge dipping down
        # when the ankle pitches/rolls; a sole-center site cannot do this.
        up = torch.zeros((self.num_envs, 3), device=self.device, dtype=gs.tc_float)
        up[:, 2] = 1.0
        heights, flatness = [], []
        for foot, vertices in enumerate(self.sole_vertices):
            local_up = quat_rotate_inverse(self.foot_quat[:, foot], up)
            lowest = (local_up @ vertices.T).amin(dim=1)
            heights.append(self.link_pos[:, self.feet_idx[foot], 2] + lowest)
            flatness.append((1.0 - local_up[:, self.sole_normal_axis[foot]].square()).clamp_min(0.0))
        self.foot_clearance = torch.stack(heights, dim=1)
        self.foot_flatness = torch.stack(flatness, dim=1)

    def _update_gait(self):
        period = self.gait["period_s"]
        self.phase = ((self.episode_length_buf.to(gs.tc_float) * self.dt / period) + self.phase_offset) % 1.0
        leg_phase = torch.stack([self.phase, (self.phase + 0.5) % 1.0], dim=1)  # [left, right]
        self.leg_phase = leg_phase
        self.stance_mask = leg_phase < self.gait["stance_ratio"]
        swing_phase = ((leg_phase - self.gait["stance_ratio"]) / (1.0 - self.gait["stance_ratio"])).clamp(0, 1)
        # Smooth lift and landing; reward follows gait intent even if both feet touch.
        self.swing_height_target = self.gait["swing_height"] * torch.sin(math.pi * swing_phase).square()
        standing = (self.commands[:, 0].abs() < 0.01) & (self.commands[:, 2].abs() < 0.04)
        self.stance_mask[standing] = True
        self.swing_height_target[standing] = 0.0

    def _update_swing_pairs(self):
        """Compare completed, phase-matched swings, never simultaneous foot heights."""
        swing = ~self.stance_mask
        started = swing & ~self.previous_swing
        # Ignore a partial initial swing when PPO randomizes episode clocks.
        full_start = self.leg_phase <= self.gait["stance_ratio"] + self.dt / self.gait["period_s"] + 1e-5
        self.swing_started |= started & full_start
        # Measure actual airborne frames for assessment and the paired timing penalty.
        self.swing_airborne_time = torch.where(started, 0.0, self.swing_airborne_time)
        self.swing_airborne_time += (swing & self.swing_started & ~self.foot_contact).to(gs.tc_float) * self.dt
        clearance = torch.where(self.foot_contact, 0.0, self.foot_clearance.clamp_min(0))
        self.swing_running_peak = torch.where(started, clearance, self.swing_running_peak)
        self.swing_running_peak = torch.where(swing, torch.maximum(self.swing_running_peak, clearance), self.swing_running_peak)
        finished = ~swing & self.previous_swing & self.swing_started
        self.completed_swing_peaks = torch.where(finished, self.swing_running_peak, self.completed_swing_peaks)
        self.completed_swing_times = torch.where(finished, self.swing_airborne_time, self.completed_swing_times)
        self.swing_pair_ready |= finished
        self.swing_started &= swing
        self.completed_swing_pair = self.swing_pair_ready.all(dim=1)
        self.completed_swing_peak_gap = (self.completed_swing_peaks[:, 0] - self.completed_swing_peaks[:, 1]).abs()
        minimum = float(self.gait["minimum_swing_clearance_m"])
        target = float(self.gait["swing_height"])
        tolerance = float(self.gait.get("peak_tolerance_m", 0.005))
        valid = ((self.completed_swing_peaks >= minimum)
                 & (self.completed_swing_peaks <= target + tolerance)
                 & (self.completed_swing_times > 0)).all(dim=1)
        self.completed_swing_pair_valid.copy_(self.completed_swing_pair & valid)
        high = self.completed_swing_peaks.amax(dim=1)
        low = self.completed_swing_peaks.amin(dim=1)
        self.completed_swing_balance.copy_(torch.where(high > 1e-6, low / high.clamp_min(1e-6), 0.0))
        self.completed_swing_time_gap.copy_((self.completed_swing_times[:, 0] - self.completed_swing_times[:, 1]).abs())
        self.completed_swing_time_relative_gap.copy_(
            (self.completed_swing_time_gap / self.completed_swing_times.mean(dim=1).clamp_min(0.1)).clamp(0.0, 1.0))
        self.swing_pair_ready[self.completed_swing_pair] = False
        self.previous_swing.copy_(swing)

    # ------------------------------------------------------------------------- #
    def _capture_diagnostics(self, tilt, fell):
        diagnostics = {
            "base_pos": self.base_pos.clone(), "base_lin_vel": self._command_velocity().clone(),
            "body_lin_vel": self.base_lin_vel.clone(), "body_yaw_rate": self.base_ang_vel[:, 2].clone(),
            "completed_swing_peaks": self.completed_swing_peaks.clone(),
            "completed_swing_pair": self.completed_swing_pair.clone(),
            "completed_swing_pair_valid": self.completed_swing_pair_valid.clone(),
            "completed_swing_peak_gap": self.completed_swing_peak_gap.clone(),
            "completed_swing_balance": self.completed_swing_balance.clone(),
            "completed_swing_times": self.completed_swing_times.clone(),
            "completed_swing_time_gap": self.completed_swing_time_gap.clone(),
            "completed_swing_time_relative_gap": self.completed_swing_time_relative_gap.clone(),
            "tilt": tilt.clone(), "fell": fell.clone(), "timeout": self.time_out_buf.clone(),
            "raw_yaw": self.raw_yaw.clone(), "heading_error": self.heading_error.clone(),
            "foot_contact": self.foot_contact.clone(),
            "foot_clearance": self.foot_clearance.clone(),
            "foot_speed": self.link_vel[:, self.feet_idx, :2].norm(dim=-1).clone(),
            "arm_leg_contact": self.arm_leg_contact.clone(),
            "arm_leg_contact_force_n": self.arm_leg_contact_force.clone(),
            "arm_leg_obstruction_time_s": self.arm_leg_obstruction_time.clone(),
            "arm_leg_obstructed": (self.arm_leg_obstruction_time >= self.arm_leg_obstruction_duration_s).clone(),
            "dof_vel": self.dof_vel.clone(), "slew_sat": self.slew_sat.clone(),
            "torque": self.robot.get_dofs_control_force(self.motor_dofs).clone(),
            "torque_limit": self.servo_torque.clone(), "speed_limit": self.servo_speed.clone(),
            "torque_sat": self.torque_sat.clone(), "speed_sat": self.speed_sat.clone(),
        }
        if self.servo_mode == "torque_speed":
            diagnostics["applied_torque"] = self.applied_torque.clone()
        return diagnostics

    def _run_physics_substeps(self):
        kernels = self.motor_kernels
        for substep in range(self.physics_steps):
            if kernels is None:
                q = self.robot.get_dofs_position(self.motor_dofs)
                qd = self.robot.get_dofs_velocity(self.motor_dofs)
                torque, clipped, speed_limited = self._servo_force(q, qd)
                self.applied_torque.copy_(torque)
                self.joint_torque_clip_count += clipped.to(gs.tc_float).sum(dim=0)
                self.joint_speed_clip_count += speed_limited.to(gs.tc_float).sum(dim=0)
                self.torque_effort += (torque / self.servo_torque).square().mean(dim=1) / self.physics_steps
                self.torque_sat |= clipped
                self.speed_sat |= speed_limited
            else:
                # Refresh q/qd after each 2 ms physics advance so derivative
                # damping and the DC motor envelope use the current state.
                kernels.servo()
            self.robot.control_dofs_force(self.applied_torque, self.motor_dofs)
            self.joint_torque_square_sum += self.applied_torque.square().sum(dim=0)
            self.joint_torque_peak_abs.copy_(torch.maximum(
                self.joint_torque_peak_abs, self.applied_torque.abs().amax(dim=0)))
            self.joint_torque_sample_count += self.num_envs
            self.scene.step()

    def step(self, actions):
        self.extras.pop("episode", None)
        self.completed_swing_pair.zero_()
        self.completed_swing_pair_valid.zero_()
        self.hard_speed_limited.zero_()
        if actions.shape != (self.num_envs, self.num_actions):
            raise ValueError(f"Expected eight leg actions shaped ({self.num_envs}, {self.num_actions}); arm actions are forbidden")
        if not torch.isfinite(actions).all():
            raise ValueError("Policy leg actions must be finite")
        self.actions = torch.clip(actions, -self.clip_actions, self.clip_actions)
        self.previous_dof_vel.copy_(self.dof_vel)
        desired = self.servo_rest.unsqueeze(0).repeat(self.num_envs, 1)
        desired[:, :LEG_JOINT_COUNT] += self.actions * self.action_scale[:LEG_JOINT_COUNT]
        desired[:, LEG_JOINT_COUNT:] = self.external_arms.commands
        desired = desired.clamp(self.target_lo, self.target_hi)
        self.motor_target_hist = torch.cat([self.motor_target_hist[:, 1:], desired.unsqueeze(1)], dim=1)
        if not self.env_cfg.get("skip_command_observer", False):
            self.observer_hist = torch.cat([self.observer_hist[:, 1:], desired.unsqueeze(1)], dim=1)
            observer_target = self.observer_hist[:, 0]
            observer_alpha = 1.0 if self.filter_tau == 0 else -math.expm1(-self.dt / self.filter_tau)
            observer_delta = observer_alpha * (observer_target - self.observer_pos)
            observer_max_step = self.target_slew_speed.unsqueeze(0) * self.dt
            observer_delta = torch.clamp(observer_delta, -observer_max_step, observer_max_step)
            self.observer_pos += observer_delta
            self.observer_vel = observer_delta / self.dt
        rows = torch.arange(self.num_envs, device=self.device)
        target = self.motor_target_hist[rows, self.max_delay - self.action_delay]
        alpha = 1.0 if self.filter_tau == 0 else -math.expm1(-self.dt / self.filter_tau)
        filtered = self.servo_target + alpha * (target - self.servo_target)
        max_step = self.target_slew_speed.unsqueeze(0) * self.dt
        delta = filtered - self.servo_target
        self.slew_sat = delta.abs() > max_step
        self.servo_target += torch.clamp(delta, -max_step, max_step)
        self._apply_pushes()
        self.torque_sat.zero_()
        self.speed_sat.zero_()
        self.torque_effort.zero_()
        if self.servo_mode == "pd":
            self.robot.control_dofs_position(self.servo_target, self.motor_dofs)
            self.scene.step()
        else:
            self._run_physics_substeps()

        self.episode_length_buf += 1
        self._update_state()
        self._update_arm_leg_contacts()
        self._update_motor_stall()
        self.dof_acc = (self.dof_vel - self.previous_dof_vel) / self.dt
        self._update_joint_sensor()
        self.imu.update(self.base_quat, self.world_lin_vel, self.base_ang_vel)
        # Sample this completed transition with the advanced gait phase and
        # command that produced it, before command resampling and auto-reset.
        self._update_gait()
        if self.clearance_pair_enabled or self.collect_diagnostics:
            self._update_swing_pairs()
        if not self.env_cfg.get("skip_training_metrics", False):
            self._accumulate_training_metrics()
        if self.fixed_command is None and not getattr(self, "external_command_scheduler", False):
            self._resample_commands((self.episode_length_buf % self.resample_steps == 0).nonzero(as_tuple=False).reshape(-1))
        if not getattr(self, "external_command_scheduler", False):
            self._update_gait()

        # termination
        tilt = torch.acos(torch.clamp(-self.projected_gravity[:, 2], -1.0, 1.0))
        self.time_out_buf = self.episode_length_buf >= self.max_episode_length
        fell = tilt > self.max_tilt
        fell |= self.base_pos[:, 2] < self.min_height
        if self.ground_support_only:
            fell |= self._nonfoot_ground_contact()
        else:
            fell |= (self.force_norm[:, self.term_idx] > self.force_thr).any(dim=1)
        arm_leg_obstructed = self.arm_leg_obstruction_time >= self.arm_leg_obstruction_duration_s
        self.fell_buf = fell
        self.reset_buf = self.fell_buf | self.time_out_buf
        self.extras["time_outs"] = self.time_out_buf.to(gs.tc_float)
        completed = self.reset_buf
        self.training_episode_count += completed.to(gs.tc_float).sum()
        self.training_episode_duration_sum_s += (self.episode_length_buf[completed].to(gs.tc_float) * self.dt).sum()
        self.training_fall_count += (fell & completed).to(gs.tc_float).sum()
        self.training_stall_count += (self.motor_stall_failure & completed).to(gs.tc_float).sum()
        self.training_arm_obstruction_count += (arm_leg_obstructed & completed).to(gs.tc_float).sum()

        self._compute_rewards()
        if self.collect_diagnostics:
            self.extras["diagnostics"] = self._capture_diagnostics(tilt, fell)

        # Publish exogenous changes before the next actor observation, so a new
        # pose is visible to the actor before its first physical movement.
        self.external_arms.advance()
        self.last_actions.copy_(self.actions)
        envs_idx = self.reset_buf.nonzero(as_tuple=False).reshape(-1)
        if len(envs_idx) > 0:
            self.reset_idx(envs_idx)
            self._update_state()
            self.contact_forces[envs_idx] = 0.0
            self.force_norm[envs_idx] = 0.0
            self.foot_contact[envs_idx] = False
            if self.ground_support_only:
                self.ground_contact_force[envs_idx] = 0.0
                self.foot_ground_force[envs_idx] = 0.0
            self._update_gait()

        self._compute_obs()
        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    # ------------------------------------------------------------------------- #
    def reset_idx(self, envs_idx, noise=True):
        if len(envs_idx) == 0:
            return
        n = len(envs_idx)
        completed_duration_s = self.episode_length_buf[envs_idx].to(gs.tc_float) * self.dt
        has_completed_episode = completed_duration_s > 0.0
        self.episode_realism_alpha[envs_idx] = self.training_realism_alpha
        replay_probability = (float(self.realism_curriculum_cfg.get("nominal_replay_probability", 0.0))
                              if self.training_curriculum_active else 0.0)
        if replay_probability > 0:
            nominal_replay = torch.rand((n,), device=self.device) < replay_probability
            self.episode_realism_alpha[envs_idx[nominal_replay]] = 0.0
            self.episode_nominal_replay[envs_idx] = nominal_replay
        else:
            self.episode_nominal_replay[envs_idx] = False
        self._randomize_actuators(envs_idx)
        dof = self.default_dof_pos.unsqueeze(0).repeat(n, 1)
        if noise and self.reset_noise > 0:
            dof[:, :LEG_JOINT_COUNT] += rand_uniform(-self.reset_noise, self.reset_noise, (n, self.num_actions), self.device)
        self.external_arms.reset(envs_idx)
        dof[:, LEG_JOINT_COUNT:] = self.external_arms.commands[envs_idx]
        self.robot.set_dofs_position(dof, self.motor_dofs, envs_idx=envs_idx, zero_velocity=True)
        self.robot.set_pos(self.base_init_pos.unsqueeze(0).repeat(n, 1), envs_idx=envs_idx, zero_velocity=True)
        self.robot.set_quat(self.base_init_quat.unsqueeze(0).repeat(n, 1), envs_idx=envs_idx, zero_velocity=True)
        self.robot.zero_all_dofs_velocity(envs_idx)

        self.actions[envs_idx] = 0.0
        self.last_actions[envs_idx] = 0.0
        reset_target = self.servo_rest.repeat(n, 1)
        reset_target[:, LEG_JOINT_COUNT:] = self.external_arms.commands[envs_idx]
        self.motor_target_hist[envs_idx] = reset_target.unsqueeze(1)
        self.servo_target[envs_idx] = reset_target
        self.servo_play_target[envs_idx] = reset_target
        measured_dof = dof[:, self.measured_joint_indices]
        self.sensor_hist[envs_idx] = measured_dof.unsqueeze(1).repeat(1, self.sensor_delay+1, 1)
        self.sensor_pos[envs_idx] = self._sample_joint_positions(measured_dof, envs_idx)
        self.observer_hist[envs_idx] = reset_target.unsqueeze(1)
        self.observer_pos[envs_idx] = reset_target
        self.observer_vel[envs_idx] = 0.0
        self.previous_dof_vel[envs_idx] = 0.0
        self.dof_acc[envs_idx] = 0.0
        self.slew_sat[envs_idx] = False
        self.torque_sat[envs_idx] = False
        self.speed_sat[envs_idx] = False
        self.applied_torque[envs_idx] = 0.0
        self.motor_stall_duration[envs_idx] = 0.0
        self.motor_stall_failure[envs_idx] = False
        self.arm_leg_obstruction_time[envs_idx] = 0.0
        self.arm_leg_contact[envs_idx] = False
        self.arm_leg_contact_force[envs_idx] = 0.0
        self.torque_effort[envs_idx] = 0.0
        self.hard_speed_limited[envs_idx] = False
        self.episode_length_buf[envs_idx] = 0
        # Exact bilateral balance at initialization; alternate starting leg at reset.
        self.phase_offset[envs_idx] = ((envs_idx + self.reset_count[envs_idx]) % 2).to(gs.tc_float) * 0.5
        self.reset_count[envs_idx] += 1
        self.heading_reference[envs_idx] = 0.0  # logical control frame is aligned with the course at reset
        self.previous_swing[envs_idx] = False
        self.swing_started[envs_idx] = False
        self.swing_running_peak[envs_idx] = 0
        self.completed_swing_peaks[envs_idx] = 0
        self.swing_pair_ready[envs_idx] = False
        self.swing_airborne_time[envs_idx] = 0
        self.completed_swing_pair[envs_idx] = False
        self.completed_swing_pair_valid[envs_idx] = False
        self.completed_swing_peak_gap[envs_idx] = 0
        self.completed_swing_balance[envs_idx] = 0
        self.completed_swing_times[envs_idx] = 0
        self.completed_swing_time_gap[envs_idx] = 0
        self.completed_swing_time_relative_gap[envs_idx] = 0
        self.imu.reset(envs_idx)
        self.lateral_origin[envs_idx] = self.base_init_pos[1]
        self.reset_buf[envs_idx] = True

        ep = {}
        if (not self.env_cfg.get("skip_episode_logging", False) and self.episode_reward_names
                and has_completed_episode.any()):
            completed_sums = self.episode_sum_matrix[:, envs_idx][:, has_completed_episode]
            durations = completed_duration_s[has_completed_episode]
            per_episode_rates = completed_sums / durations.unsqueeze(0)
            if self.rew_cfg.get("termination_is_event", False) and "termination" in self.episode_reward_names:
                # Event termination is already an episode-level penalty; report its mean event value.
                term_row = self.episode_reward_names.index("termination")
                per_episode_rates[term_row] = completed_sums[term_row]
            means = per_episode_rates.mean(dim=1).detach()
            ep.update(("rew_" + name, means[i])
                      for i, name in enumerate(self.episode_reward_names))
        self.episode_sum_matrix[:, envs_idx] = 0.0
        self.extras["episode"] = ep
        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        self._update_state()
        self._update_gait()
        self._compute_obs()
        self.extras.pop("episode", None)
        return self.obs_buf, None

    def _accumulate_training_metrics(self):
        """Accumulate rollout speed and intended-swing sole-clearance statistics on device."""
        forward_velocity = self._command_velocity()[:, 0]
        forward_command = self.commands[:, 0]
        self.training_velocity_sums += torch.stack((
            forward_velocity.mean(), forward_command.mean(),
            (forward_velocity - forward_command).abs().mean(),
        ))
        self.training_velocity_steps += 1

        swing = ~self.stance_mask
        clearance = self.foot_clearance.clamp_min(0.0)
        swing_float = swing.to(gs.tc_float)
        self.training_swing_clearance_sum += (clearance * swing_float).sum(dim=0)
        self.training_swing_samples += swing_float.sum(dim=0)
        self.training_swing_peaks.copy_(torch.maximum(
            self.training_swing_peaks, torch.where(swing, clearance, torch.zeros_like(clearance))))
        self.training_swing_seen |= swing
        f = gs.tc_float
        self.training_servo_sums += torch.stack((
            self.slew_sat.to(f).mean() * 100.0,
            self.torque_sat.to(f).mean() * 100.0,
            self.speed_sat.to(f).mean() * 100.0,
            (self.dof_vel.abs() > self.servo_speed).to(f).mean() * 100.0,
        ))

        self.course_metric_sums += torch.stack((self.heading_error.abs().mean(),
                                                (self.base_pos[:, 1]-self.lateral_origin).abs().mean()))
        paired = self.completed_swing_pair.to(f)
        self.completed_peak_sums[:2] += (self.completed_swing_peaks * paired[:, None]).sum(dim=0)
        self.completed_peak_sums[2] += (self.completed_swing_peak_gap * paired).sum()
        attempts = paired.sum()
        self.completed_pair_count += attempts
        self.completed_valid_pair_count += (self.completed_swing_pair_valid & self.completed_swing_pair).sum()
        self.completed_pair_balance_sum += (self.completed_swing_balance * paired).sum()
        self.completed_time_gap_sum += (self.completed_swing_time_gap * paired).sum()
        self.completed_time_relative_gap_sum += (self.completed_swing_time_relative_gap * paired).sum()

    @torch.inference_mode()
    def consume_training_metrics(self):
        """Return the current rollout's speed and clearance metrics, then clear accumulators."""
        speed = self.training_velocity_sums / max(self.training_velocity_steps, 1)
        clearance_mean = self.training_swing_clearance_sum / self.training_swing_samples.clamp_min(1.0)
        seen = self.training_swing_seen.to(gs.tc_float)
        peak_mean = (self.training_swing_peaks * seen).sum(dim=0) / seen.sum(dim=0).clamp_min(1.0)
        servo = self.training_servo_sums / max(self.training_velocity_steps, 1)
        packed = torch.stack((
            speed[0], speed[1], speed[2],
            clearance_mean[0], clearance_mean[1],
            peak_mean[0], peak_mean[1],
            self.training_swing_samples[0], self.training_swing_samples[1],
            servo[0], servo[1], servo[2], servo[3],
        )).tolist()
        names = (
            "mean_forward_velocity_m_s", "mean_command_forward_velocity_m_s",
            "forward_velocity_mae_m_s", "mean_left_swing_clearance_m",
            "mean_right_swing_clearance_m", "mean_left_peak_swing_clearance_m",
            "mean_right_peak_swing_clearance_m", "target_slew_saturation_pct",
            "torque_saturation_pct", "drive_speed_saturation_pct", "joint_overspeed_pct",
        )
        metrics = dict(zip(names[:7], packed[:7]))
        metrics.update(zip(names[7:], packed[9:13]))
        if self.training_velocity_steps == 0:
            for name in names[:3]:
                metrics[name] = None
        if packed[7] == 0:
            metrics["mean_left_swing_clearance_m"] = None
            metrics["mean_left_peak_swing_clearance_m"] = None
        if packed[8] == 0:
            metrics["mean_right_swing_clearance_m"] = None
            metrics["mean_right_peak_swing_clearance_m"] = None
        if self.training_velocity_steps == 0:
            for name in names[7:]:
                metrics[name] = None
        if self.servo_mode != "torque_speed":
            metrics["torque_saturation_pct"] = None
            metrics["drive_speed_saturation_pct"] = None
            metrics["joint_overspeed_pct"] = None
        metrics["target_peak_swing_clearance_m"] = float(self.gait["swing_height"])
        episode_count = float(self.training_episode_count.item())
        metrics["episode_completion_count"] = episode_count
        metrics["episode_mean_duration_s"] = (float(self.training_episode_duration_sum_s.item()) / episode_count
                                               if episode_count else None)
        metrics["fall_event_count"] = float(self.training_fall_count.item())
        metrics["stall_event_count"] = float(self.training_stall_count.item())
        metrics["arm_leg_obstruction_event_count"] = float(self.training_arm_obstruction_count.item())

        course = (self.course_metric_sums / max(self.training_velocity_steps, 1)).tolist()
        metrics.update(zip(("heading_error_mean_abs_rad", "lateral_excursion_mean_abs_m"), course))
        pair_count = self.completed_pair_count.item()
        peaks = (self.completed_peak_sums / max(pair_count, 1)).tolist()
        metrics.update(zip(("mean_left_completed_swing_peak_m", "mean_right_completed_swing_peak_m",
                            "paired_swing_peak_gap_m"), peaks if pair_count else (None, None, None)))
        metrics["completed_pair_count"] = int(pair_count)
        metrics["valid_pair_completion_ratio"] = (
            float(self.completed_valid_pair_count.item() / max(pair_count, 1)) if pair_count else None)
        metrics["paired_swing_balance_ratio"] = (
            float(self.completed_pair_balance_sum.item() / max(pair_count, 1)) if pair_count else None)
        metrics["paired_swing_time_gap_s"] = (
            float(self.completed_time_gap_sum.item() / max(pair_count, 1)) if pair_count else None)
        metrics["paired_swing_time_gap_fraction"] = (
            float(self.completed_time_relative_gap_sum.item() / max(pair_count, 1)) if pair_count else None)
        if self.servo_mode == "torque_speed" and self.joint_torque_sample_count:
            joint_rms = (self.joint_torque_square_sum / self.joint_torque_sample_count).sqrt().tolist()
            joint_peak = self.joint_torque_peak_abs.tolist()
            for index, joint_name in enumerate(self.joint_names):
                key = joint_name.replace("-", "_")
                group = "leg" if index < LEG_JOINT_COUNT else "arm"
                metric_prefix = f"motor/{joint_name}/"
                metrics[metric_prefix + "rms_torque_nm"] = float(joint_rms[index])
                metrics[metric_prefix + "peak_torque_nm"] = float(joint_peak[index])
                metrics[metric_prefix + "torque_clip_pct"] = float(
                    100.0*self.joint_torque_clip_count[index].item()/self.joint_torque_sample_count)
                metrics[metric_prefix + "speed_clip_pct"] = float(
                    100.0*self.joint_speed_clip_count[index].item()/self.joint_torque_sample_count)
                metrics[metric_prefix + "stall_peak_duration_s"] = float(self.joint_stall_duration_peak[index].item())
                metrics[metric_prefix + "stall_event_count"] = float(self.joint_stall_event_count[index].item())
                metrics[f"{group}_{key}_torque_rms_nm"] = float(joint_rms[index])
                metrics[f"{group}_{key}_torque_peak_nm"] = float(joint_peak[index])
                metrics[f"{group}_{key}_torque_clip_pct"] = float(
                    100.0*self.joint_torque_clip_count[index].item()/self.joint_torque_sample_count)
                metrics[f"{group}_{key}_speed_clip_pct"] = float(
                    100.0*self.joint_speed_clip_count[index].item()/self.joint_torque_sample_count)
        else:
            for index, joint_name in enumerate(self.joint_names):
                key = joint_name.replace("-", "_")
                group = "leg" if index < LEG_JOINT_COUNT else "arm"
                metric_prefix = f"motor/{joint_name}/"
                metrics[metric_prefix + "rms_torque_nm"] = None
                metrics[metric_prefix + "peak_torque_nm"] = None
                metrics[metric_prefix + "torque_clip_pct"] = None
                metrics[metric_prefix + "speed_clip_pct"] = None
                metrics[metric_prefix + "stall_peak_duration_s"] = float(self.joint_stall_duration_peak[index].item())
                metrics[metric_prefix + "stall_event_count"] = float(self.joint_stall_event_count[index].item())
                metrics[f"{group}_{key}_torque_rms_nm"] = None
                metrics[f"{group}_{key}_torque_peak_nm"] = None
                metrics[f"{group}_{key}_torque_clip_pct"] = None
                metrics[f"{group}_{key}_speed_clip_pct"] = None
        self.joint_torque_square_sum.zero_()
        self.joint_torque_peak_abs.zero_()
        self.joint_torque_clip_count.zero_()
        self.joint_speed_clip_count.zero_()
        self.joint_stall_duration_peak.zero_()
        self.joint_stall_event_count.zero_()
        self.joint_torque_sample_count = 0
        self.course_metric_sums.zero_()
        self.completed_peak_sums.zero_()
        self.completed_pair_count.zero_()
        self.completed_valid_pair_count.zero_()
        self.completed_pair_balance_sum.zero_()
        self.completed_time_gap_sum.zero_()
        self.completed_time_relative_gap_sum.zero_()
        self.training_velocity_sums.zero_()
        self.training_velocity_steps = 0
        self.training_episode_count.zero_()
        self.training_episode_duration_sum_s.zero_()
        self.training_fall_count.zero_()
        self.training_stall_count.zero_()
        self.training_arm_obstruction_count.zero_()
        self.training_swing_clearance_sum.zero_()
        self.training_swing_samples.zero_()
        self.training_swing_peaks.zero_()
        self.training_swing_seen.zero_()
        self.training_servo_sums.zero_()
        return metrics

    # ------------------------------------------------------------------------- #
    def _compute_obs(self, advance_history=True):
        s = self.obs_scales
        clock = torch.stack((torch.sin(2*math.pi*self.phase), torch.cos(2*math.pi*self.phase)), dim=1)
        command = torch.stack((self.commands[:, 0], self.commands[:, 2]), dim=1) * self.cmd_scale[:2]
        arm_commands = (self.external_arms.commands-self.default_dof_pos[LEG_JOINT_COUNT:]) * s["dof_pos"]
        clean = (self.base_ang_vel*s["ang_vel"], self.projected_gravity, command,
                 (self.dof_pos-self.default_dof_pos)*s["dof_pos"],
                 self.dof_vel*s["dof_vel"], self.last_actions, arm_commands, clock)
        # Leg positions come only from their delayed/noisy potentiometers. Arm positions
        # come only from external command history; no physical arm state is sampled.
        position = self.obs_position_buf
        position.copy_(self.observer_pos)
        position[:, self.measured_joint_indices] = self.sensor_pos
        std = float(self.noise_cfg.get("dof_pos", 0.0))
        if std > 0 and self.command_estimated_joint_indices:
            arm_index = self.command_estimated_joint_indices
            realism = self.episode_realism_alpha.unsqueeze(1)
            position[:, arm_index] += torch.randn_like(position[:, arm_index]) * std * realism
        actor_parts = (self.imu.gyro*s["ang_vel"], self.imu.gravity, command,
                       (position-self.default_dof_pos)*s["dof_pos"],
                       self.observer_vel*s["dof_vel"], self.last_actions, arm_commands, clock)
        critic_parts = (*clean, self._command_velocity()*s["lin_vel"],
                        self.foot_contact.to(gs.tc_float),
                        (self.base_pos[:, 1]-self.lateral_origin).unsqueeze(1),
                        self.heading_error.unsqueeze(1))
        # Return fresh output tensors: PPO retains the pre-action observations
        # by reference until its post-step transition storage copy completes.
        frame = torch.cat(actor_parts, dim=1)
        if frame.shape[1] != 46:
            raise RuntimeError(f"V7 actor frame contract expected 46, got {frame.shape[1]}")
        self.frame_obs_buf = frame
        fresh = self.episode_length_buf == 0
        if fresh.any():
            self.obs_history[fresh] = frame[fresh].unsqueeze(1).expand(-1, 5, -1)
        continuing = ~fresh
        if continuing.any():
            if advance_history:
                self.obs_history[continuing, :-1] = self.obs_history[continuing, 1:].clone()
            self.obs_history[continuing, -1] = frame[continuing]
        self.obs_buf = self.obs_history.reshape(self.num_envs, 230).clone()
        self.critic_buf = torch.cat(critic_parts, dim=1)
        self.extras["observations"]["critic"] = self.critic_buf

    def set_external_arm_commands(self, targets, envs_idx=None):
        """Accept Bluetooth/external joint targets in motor-order radians.

        Random sampling stops for selected environments until reset or an
        explicit resume_external_arm_sampling call. Physics still uses the same
        delay, slew, torque and speed model; these are never actor actions.
        """
        self.external_arms.set_commands(targets, envs_idx)
        self._compute_obs(advance_history=False)

    def resume_external_arm_sampling(self, envs_idx=None):
        self.external_arms.resume_sampling(envs_idx)

    def get_observations(self):
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    # ------------------------------------------------------------------------- #
    def _compute_rewards(self):
        self.rew_buf[:] = 0.0
        for name, scale in self.reward_scales.items():
            if name in ("torques", "feet_swing_time_symmetry"):
                scale *= getattr(self, "reward_penalty_alpha", 1.0)
            r = self._reward_fns[name]() * scale
            self.rew_buf += r
            self.episode_sums[name] += r

    def _command_velocity(self):
        return self.reference_lin_vel if self.command_reference_frame == "reset_heading" else self.base_lin_vel

    def _reward_feet_swing_symmetry(self):
        target = float(self.gait["swing_height"])
        gap = (self.completed_swing_peak_gap / target).clamp_max(3).square()
        return gap * self.completed_swing_pair * (float(self.gait["period_s"]) / self.dt)

    def _reward_feet_swing_time_symmetry(self):
        # Use every completed pair, including low-lift pairs, to avoid a dead zone.
        # The existing nonzero lift targets prevent stillness from earning a good gait.
        return (self.completed_swing_time_relative_gap.square() * self.completed_swing_pair
                * (float(self.gait["period_s"]) / self.dt))

    def _reward_feet_swing_peak(self):
        # Same nonzero target for BOTH feet prevents a zero-lift symmetry solution.
        target = float(self.gait["swing_height"])
        error = ((self.completed_swing_peaks-target) / target).clamp(-3, 3).square().mean(dim=1)
        return error * self.completed_swing_pair * (float(self.gait["period_s"]) / self.dt)

    def _reward_forward_progress(self):
        # Signed progress cannot turn equal forward/backward buzzing into travel.
        desired = self.commands[:, 0]
        ratio = self._command_velocity()[:, 0] / desired.clamp_min(1e-6)
        return torch.where(desired > 0, ratio.clamp_max(1.0), torch.zeros_like(ratio))

    def _bounded_course_error(self, error, tolerance_key, default):
        squared = error.square()
        if not self.rew_cfg.get("bounded_course_errors", False):
            return squared
        tolerance = float(self.rew_cfg.get(tolerance_key, default))
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(tolerance_key + " must be finite and positive")
        return squared / (squared + tolerance * tolerance)

    def _reward_lin_vel_y(self):
        return self._bounded_course_error(self._command_velocity()[:, 1],
                                          "lateral_velocity_tolerance_m_s", 0.10)

    def _reward_heading(self):
        return self._bounded_course_error(self.heading_error, "heading_tolerance_rad", 0.35)

    def _reward_lateral_drift(self):
        offset = self.base_pos[:, 1]-self.lateral_origin
        if "lateral_position_tolerance_m" in self.rew_cfg:
            return self._bounded_course_error(offset, "lateral_position_tolerance_m", 0.05)
        return offset.square()

    def _reward_tracking_lin_vel(self):
        err = torch.sum((self.commands[:, :2] - self._command_velocity()[:, :2]) ** 2, dim=1)
        return torch.exp(-err / self.rew_cfg["tracking_sigma_lin"])

    def _reward_tracking_ang_vel(self):
        err = (self.commands[:, 2] - self.base_ang_vel[:, 2]) ** 2
        return torch.exp(-err / self.rew_cfg["tracking_sigma_ang"])

    def _reward_contact_match(self):
        return (self.foot_contact == self.stance_mask).to(gs.tc_float).mean(dim=1)

    def _reward_contact_load(self):
        if not self.ground_support_only:
            raise ValueError("contact_load reward requires ground_support_only=True")
        sigma = float(self.rew_cfg.get("tracking_sigma_contact", 0.3))
        if sigma <= 0:
            raise ValueError("tracking_sigma_contact must be positive")
        forces = self.foot_ground_force.clamp_min(0.0)
        total = forces.sum(dim=1)
        ratio = forces / total.clamp_min(1e-6).unsqueeze(1)
        stance = self.stance_mask.to(gs.tc_float)
        target = stance / stance.sum(dim=1, keepdim=True).clamp_min(1.0)
        error = (ratio-target).square().sum(dim=1)
        score = torch.exp(-error/sigma)
        return torch.where(total >= 1.0, score, torch.zeros_like(score))

    def _reward_contact_load_error(self):
        if not self.ground_support_only:
            raise ValueError("contact_load_error requires ground_support_only=True")
        forces = self.foot_ground_force.clamp_min(0.0)
        total = forces.sum(dim=1)
        ratio = forces / total.clamp_min(1e-6).unsqueeze(1)
        stance = self.stance_mask.to(gs.tc_float)
        target = stance / stance.sum(dim=1, keepdim=True).clamp_min(1.0)
        error = (ratio-target).square().sum(dim=1)
        return torch.where(total >= 1.0, error, torch.full_like(error, 2.0))

    def _reward_lin_vel_z(self):
        return self.base_lin_vel[:, 2] ** 2

    def _reward_ang_vel_xy(self):
        return torch.sum(self.base_ang_vel[:, :2] ** 2, dim=1)

    def _reward_orientation(self):
        return torch.sum(self.projected_gravity[:, :2] ** 2, dim=1)

    def _reward_base_height(self):
        return (self.base_pos[:, 2] - self.rew_cfg["base_height_target"]) ** 2

    def _reward_action_rate(self):
        return torch.sum((self.last_actions - self.actions) ** 2, dim=1)

    def _reward_torques(self):
        if self.servo_mode != "torque_speed":
            raise ValueError("torques reward requires torque_speed mode")
        return self.torque_effort

    def _reward_dof_acc(self):
        return self.dof_acc.square().mean(dim=1)

    def _reward_ankle_roll_deviation(self):
        return torch.sum(torch.abs(self.dof_pos[:, self.ankle_roll_idx] - self.default_dof_pos[self.ankle_roll_idx]), dim=1)

    def _reward_feet_flat(self):
        # sin^2(tilt) of each sole from horizontal; zero for a flat foot, grows when a sole edge digs in.
        weights = self.stance_mask.to(gs.tc_float) if self.gait.get("feet_flat_stance_only", False) else 1.0
        return (self.foot_flatness * weights).sum(dim=1)

    def _reward_feet_swing_clearance(self):
        minimum = float(self.gait.get("minimum_swing_clearance_m", 0.015))
        if not 0 < minimum <= self.gait["swing_height"]:
            raise ValueError("minimum_swing_clearance_m must be positive and no higher than swing_height")
        intended_swing = (~self.stance_mask) & (self.swing_height_target >= minimum)
        deficit = ((minimum - self.foot_clearance.clamp_min(0.0)).clamp_min(0.0) / minimum).square()
        return (deficit * intended_swing.to(gs.tc_float)).sum(dim=1)

    def _reward_feet_swing_height(self):
        clearance = self.foot_clearance
        if self.gait.get("swing_height_mode", "phase") == "legacy_contact":
            # Explicit compatibility mode for reproducing old saved-run rewards.
            err = (clearance - self.gait["swing_height"]) ** 2
            return torch.sum(err * (~self.foot_contact).to(gs.tc_float), dim=1)
        err = (clearance - self.swing_height_target) ** 2
        return torch.sum(err * (~self.stance_mask).to(gs.tc_float), dim=1)

    def _reward_feet_slip(self):
        v = self.link_vel[:, self.feet_idx, :2].norm(dim=-1)
        return torch.sum(v * self.foot_contact.to(gs.tc_float), dim=1)

    def _reward_unwanted_contacts(self):
        return torch.sum((self.force_norm[:, self.pen_idx] > self.force_thr).to(gs.tc_float), dim=1)

    def _reward_termination(self):
        return (self.fell_buf & ~self.time_out_buf).to(gs.tc_float)
