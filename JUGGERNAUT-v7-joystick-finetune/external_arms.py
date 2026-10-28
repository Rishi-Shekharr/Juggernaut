"""Externally commanded shoulder/mandible poses inside a guarded envelope."""
import math
import torch

ARM_NAMES = ("left_shoulder", "left_mandible", "right_shoulder", "right_mandible")


def _bounds(cfg):
    arms = cfg["env"]["external_arms"]
    source = arms.get("safe_envelope_deg") or arms.get("pose_ranges_deg", {})
    result = {}
    for name in ARM_NAMES:
        pair = source.get(name)
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"external_arms.safe_envelope_deg.{name} must contain two degree bounds")
        lo, hi = map(float, pair)
        if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
            raise ValueError(f"external_arms.safe_envelope_deg.{name} must be finite and ordered")
        result[name] = (lo, hi)
    return result


def validate_external_arms(cfg, joints_cfg):
    arms = cfg["env"].get("external_arms", {})
    if arms.get("control") != "external" or arms.get("joint_names") != list(ARM_NAMES):
        raise ValueError("external_arms requires external control and the four ordered shoulder/mandible joints")
    bounds = _bounds(cfg)
    for joint in joints_cfg.get("joints", ())[8:]:
        name = joint["name"]
        lo, hi = bounds[name]
        physical = [math.degrees(float(v)) for v in joint["target_limit"]]
        if lo < physical[0] or hi > physical[1]:
            raise ValueError(f"external_arms.safe_envelope_deg.{name} exceeds physical joint travel")
        for key in ("reset_ranges_deg", "pose_ranges_deg"):
            pair = arms.get(key, {}).get(name, [lo, hi])
            if len(pair) != 2 or float(pair[0]) < lo or float(pair[1]) > hi or float(pair[0]) > float(pair[1]):
                raise ValueError(f"external_arms.{key}.{name} must stay inside the safe envelope")
    mode = arms.get("mode", "fixed")
    if mode not in ("fixed", "episode", "moving"):
        raise ValueError("external_arms.mode must be fixed, episode, or moving")
    command_speed = float(arms.get("command_speed_rad_s", 0.35))
    obstruction_force = float(arms.get("obstruction_force_n", 1.0))
    obstruction_duration = float(arms.get("obstruction_duration_s", 0.10))
    if not math.isfinite(command_speed) or command_speed <= 0:
        raise ValueError("external_arms.command_speed_rad_s must be finite and positive")
    if not math.isfinite(obstruction_force) or obstruction_force <= 0:
        raise ValueError("external_arms.obstruction_force_n must be finite and positive")
    if not math.isfinite(obstruction_duration) or obstruction_duration <= 0:
        raise ValueError("external_arms.obstruction_duration_s must be finite and positive")
    for key, minimum in (("hold_s", 5.0), ("reset_hold_s", 1.0)):
        pair = arms.get(key, [minimum, minimum])
        if len(pair) != 2 or float(pair[0]) < minimum or float(pair[1]) < float(pair[0]):
            raise ValueError(f"external_arms.{key} must be ordered and at least {minimum:g} seconds")
    if float(arms.get("settling_margin_s", 0.5)) < 0:
        raise ValueError("external_arms.settling_margin_s must be nonnegative")


class ExternalArmCommands:
    """Sampled and manual targets share one safe clamp and rate-limited command path."""
    def __init__(self, num_envs, cfg, device, dtype, dt, target_lo, target_hi):
        self.cfg = cfg["env"]["external_arms"]
        self.device, self.dtype, self.dt = device, dtype, float(dt)
        self.mode = self.cfg.get("mode", "fixed")
        self.curriculum = self.cfg.get("curriculum", {})
        self.pose_final_alpha = float(self.curriculum.get("final_alpha", 1.0))
        # Standalone evaluation sees the full configured pose range. Training
        # explicitly selects its curriculum alpha before resetting episodes.
        self.pose_alpha = self.pose_final_alpha
        self.names = ARM_NAMES
        self.lo = torch.tensor([math.radians(_bounds(cfg)[n][0]) for n in ARM_NAMES], device=device, dtype=dtype)
        self.hi = torch.tensor([math.radians(_bounds(cfg)[n][1]) for n in ARM_NAMES], device=device, dtype=dtype)
        self.center = torch.tensor([math.radians(float(self.cfg.get("reset_pose_deg", {}).get(n, 0.0))) for n in ARM_NAMES], device=device, dtype=dtype)
        self.center = self.center.clamp(self.lo, self.hi)
        self.reset_lo, self.reset_hi = self._ranges("reset_ranges_deg")
        self.pose_lo, self.pose_hi = self._ranges("pose_ranges_deg")
        self.commands = self.center.repeat(num_envs, 1)
        self.goal = self.commands.clone()
        self.remaining_steps = torch.zeros(num_envs, device=device, dtype=torch.long)
        self.manual = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.rows = torch.arange(num_envs, device=device)
        servo = cfg["env"].get("servo", {})
        command_speed = float(self.cfg.get("command_speed_rad_s", 0.35))
        if not math.isfinite(command_speed) or command_speed <= 0:
            raise ValueError("external_arms.command_speed_rad_s must be finite and positive")
        role_speeds = [command_speed for _ in ARM_NAMES]
        self.rate = torch.tensor(role_speeds, device=device, dtype=dtype) * self.dt
        self.margin = float(self.cfg.get("settling_margin_s", 0.5))

    def _ranges(self, key):
        ranges = self.cfg.get(key, {})
        values = []
        for name in ARM_NAMES:
            pair = ranges.get(name, [math.degrees(float(self.center[ARM_NAMES.index(name)]))]*2)
            values.append([math.radians(float(v)) for v in pair])
        tensor = torch.tensor(values, device=self.device, dtype=self.dtype)
        return tensor[:, 0], tensor[:, 1]

    def _sample(self, lo, hi, n):
        return lo + torch.rand((n, 4), device=self.device, dtype=self.dtype) * (hi-lo)

    def _hold_steps(self, n, key="hold_s"):
        lo, hi = map(float, self.cfg.get(key, [5.0, 8.0]))
        return torch.ceil((lo + torch.rand(n, device=self.device, dtype=self.dtype)*(hi-lo))/self.dt).long()

    def set_training_iteration(self, iteration):
        self.iteration = int(iteration)
        warmup = int(self.curriculum.get("warmup_iterations", 0))
        ramp = max(1, int(self.curriculum.get("ramp_iterations", 1)))
        progress = min(1.0, max(0.0, (int(iteration)-warmup)/ramp))
        self.pose_alpha = float(self.curriculum.get("initial_alpha", 0.0)) + progress*(self.pose_final_alpha-float(self.curriculum.get("initial_alpha", 0.0)))

    def reset(self, envs_idx):
        self.manual[envs_idx] = False
        if self.mode in ("fixed", "moving"):
            value = self.center.expand(len(envs_idx), -1)
        else:
            lo = self.center + self.pose_alpha*(self.pose_lo-self.center)
            hi = self.center + self.pose_alpha*(self.pose_hi-self.center)
            value = self._sample(lo, hi, len(envs_idx))
        self.commands[envs_idx] = value.clamp(self.lo, self.hi)
        self.goal[envs_idx] = self.commands[envs_idx]
        self.remaining_steps[envs_idx] = self._hold_steps(len(envs_idx), "reset_hold_s")

    def advance(self):
        step = self.rate.unsqueeze(0)
        delta = (self.goal-self.commands).clamp(-step, step)
        self.commands.add_(delta)
        self.remaining_steps -= 1
        if self.mode != "moving":
            return
        due = self.rows[(self.remaining_steps <= 0) & ~self.manual & (self.commands-self.goal).abs().amax(dim=1).lt(1e-5)]
        if len(due):
            lo = self.center + self.pose_alpha*(self.pose_lo-self.center)
            hi = self.center + self.pose_alpha*(self.pose_hi-self.center)
            self.goal[due] = self._sample(lo, hi, len(due)).clamp(self.lo, self.hi)
            travel = (self.goal[due]-self.commands[due]).abs().amax(dim=1) / self.rate.amin().clamp_min(1e-6) * self.dt
            self.remaining_steps[due] = self._hold_steps(len(due)) + torch.ceil((travel+self.margin)/self.dt).long()

    def set_commands(self, targets, envs_idx=None):
        rows = self.rows if envs_idx is None else torch.as_tensor(envs_idx, device=self.device, dtype=torch.long).reshape(-1)
        value = torch.as_tensor(targets, device=self.device, dtype=self.dtype)
        if value.shape == (4,): value = value.unsqueeze(0).expand(len(rows), -1)
        if value.shape != (len(rows), 4) or not torch.isfinite(value).all():
            raise ValueError("External arm targets must be four finite radians per selected environment")
        self.goal[rows] = value.clamp(self.lo, self.hi)
        self.manual[rows] = True

    def resume_sampling(self, envs_idx=None):
        rows = self.rows if envs_idx is None else torch.as_tensor(envs_idx, device=self.device, dtype=torch.long).reshape(-1)
        self.manual[rows] = False
        self.remaining_steps[rows] = 0
