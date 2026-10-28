"""Batched six-axis IMU model for JUGGERNAUT V7.

The six-axis gyro/accelerometer model has no magnetic heading or foot switches.
"""
import math
import torch


def rotate_inverse(q, v):
    w, xyz = q[..., :1], q[..., 1:]
    return v*(2*w*w-1)-2*w*torch.cross(xyz, v, dim=-1)+2*xyz*(xyz*v).sum(-1, keepdim=True)


class SixAxisIMU:
    """Gyro/accelerometer tilt fusion. No yaw reference or magnetic input exists."""
    def __init__(self, n, cfg, dt, device, dtype):
        self.cfg, self.dt, self.device, self.dtype = cfg, dt, device, dtype
        self.delay = int(cfg.get("delay_steps", 1))
        if self.delay < 0:
            raise ValueError("imu.delay_steps must be nonnegative")
        for key in ("gyro_noise_std_rad_s", "gyro_bias_std_rad_s", "gyro_bias_walk_std_rad_s_sqrt_s",
                    "accel_noise_std_m_s2", "accel_bias_std_m_s2"):
            value = float(cfg.get(key, 0.0))
            if not math.isfinite(value) or value < 0:
                raise ValueError("imu."+key+" must be finite and nonnegative")
        for key in ("accel_full_scale_m_s2", "accel_quantization_m_s2", "tilt_filter_tau_s", "accel_gate_fraction"):
            value = float(cfg[key])
            if not math.isfinite(value) or value <= 0:
                raise ValueError("imu."+key+" must be finite and positive")
        self.gyro_bias = torch.zeros((n, 3), device=device, dtype=dtype)
        self.accel_bias = torch.zeros_like(self.gyro_bias)
        self.previous_velocity = torch.zeros_like(self.gyro_bias)
        self.gyro = torch.zeros_like(self.gyro_bias)
        self.gravity = torch.zeros_like(self.gyro_bias)
        self.gravity[:, 2] = -1
        self.history = torch.zeros((n, self.delay+1, 6), device=device, dtype=dtype)

    def _quantize(self, value):
        step, limit = float(self.cfg["accel_quantization_m_s2"]), float(self.cfg["accel_full_scale_m_s2"])
        return (value.clamp(-limit, limit)/step).round()*step

    def reset(self, idx):
        def noise(key):
            return torch.randn((len(idx), 3), device=self.device, dtype=self.dtype)*float(self.cfg.get(key, 0))
        self.gyro_bias[idx] = noise("gyro_bias_std_rad_s")
        self.accel_bias[idx] = noise("accel_bias_std_m_s2")
        self.previous_velocity[idx] = 0
        gyro = self.gyro_bias[idx]+noise("gyro_noise_std_rad_s")
        accel = self.accel_bias[idx]+noise("accel_noise_std_m_s2")
        accel[:, 2] += 9.80665
        accel = self._quantize(accel)
        self.gravity[idx] = -accel/accel.norm(dim=1, keepdim=True).clamp_min(1e-6)
        self.gyro[idx] = gyro
        sample = torch.cat((gyro, accel), dim=1)
        self.history[idx] = sample[:, None, :].expand(-1, self.delay+1, -1)

    def update(self, quat, world_velocity, body_omega):
        c, dt = self.cfg, self.dt
        self.gyro_bias += torch.randn_like(self.gyro_bias)*float(c.get("gyro_bias_walk_std_rad_s_sqrt_s", 0))*math.sqrt(dt)
        gyro = body_omega+self.gyro_bias+torch.randn_like(body_omega)*float(c.get("gyro_noise_std_rad_s", 0))
        acceleration = (world_velocity-self.previous_velocity)/dt
        self.previous_velocity.copy_(world_velocity)
        acceleration[:, 2] += 9.80665
        accel = rotate_inverse(quat, acceleration)+self.accel_bias
        accel += torch.randn_like(accel)*float(c.get("accel_noise_std_m_s2", 0))
        sample = torch.cat((gyro, self._quantize(accel)), dim=1)
        self.history.copy_(torch.cat((self.history[:, 1:], sample[:, None, :]), dim=1))
        delayed = self.history[:, 0]
        self.gyro.copy_(delayed[:, :3])
        force = delayed[:, 3:]
        prediction = self.gravity-torch.cross(self.gyro, self.gravity, dim=1)*dt
        prediction /= prediction.norm(dim=1, keepdim=True).clamp_min(1e-6)
        norm = force.norm(dim=1, keepdim=True)
        valid = (norm-9.80665).abs() <= float(c["accel_gate_fraction"])*9.80665
        alpha = -math.expm1(-dt/float(c["tilt_filter_tau_s"]))
        tilt = prediction+alpha*(-force/norm.clamp_min(1e-6)-prediction)*valid
        self.gravity.copy_(tilt/tilt.norm(dim=1, keepdim=True).clamp_min(1e-6))
