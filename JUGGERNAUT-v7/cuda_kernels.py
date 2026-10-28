"""Fused CUDA servo execution, adapted from 4dof.

Imported only for the requested CUDA backend. All arrays alias owned PyTorch
buffers and launches use its CURRENT CUDA stream. Physics still advances through
Genesis's public API at every original 2 ms substep.
"""
import torch
import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _servo_step(
    q: wp.array2d(dtype=float), qd: wp.array2d(dtype=float),
    target: wp.array2d(dtype=float), kp: wp.array2d(dtype=float),
    kd: wp.array2d(dtype=float), deadband: wp.array2d(dtype=float),
    backlash: wp.array2d(dtype=float), play_target: wp.array2d(dtype=float),
    peak_torque: wp.array2d(dtype=float), no_load_speed: wp.array2d(dtype=float),
    torque: wp.array2d(dtype=float), torque_sat: wp.array2d(dtype=wp.bool),
    speed_sat: wp.array2d(dtype=wp.bool), torque_clip_count: wp.array(dtype=float),
    speed_clip_count: wp.array(dtype=float), effort: wp.array(dtype=float),
    substeps: float, num_joints: int,
):
    i = wp.tid()
    normalized_effort = float(0.0)
    for j in range(num_joints):
        # Same stateful play as the Torch path; width is TOTAL reversal travel.
        half_width = 0.5*backlash[i, j]
        effective = wp.clamp(play_target[i, j], target[i, j]-half_width, target[i, j]+half_width)
        play_target[i, j] = effective
        error = effective-q[i, j]
        error = wp.sign(error)*wp.max(wp.abs(error)-deadband[i, j], 0.0)
        requested = kp[i, j]*error-kd[i, j]*qd[i, j]
        driving = requested*qd[i, j] >= 0.0
        fraction = wp.clamp(1.0-wp.abs(qd[i, j])/wp.max(no_load_speed[i, j], 1.0e-6), 0.0, 1.0)
        limit = peak_torque[i, j]
        if driving:
            limit = limit*fraction
        actual = wp.clamp(requested, -limit, limit)
        clipped = wp.abs(requested) > limit
        torque[i, j] = actual
        torque_sat[i, j] = torque_sat[i, j] or clipped
        speed_clipped = clipped and driving and wp.abs(qd[i, j]) > 0.0
        speed_sat[i, j] = speed_sat[i, j] or speed_clipped
        if clipped:
            wp.atomic_add(torque_clip_count, j, 1.0)
        if speed_clipped:
            wp.atomic_add(speed_clip_count, j, 1.0)
        ratio = actual/peak_torque[i, j]
        normalized_effort = normalized_effort+ratio*ratio
    effort[i] = effort[i]+normalized_effort/(float(num_joints)*substeps)


def view(tensor):
    if tensor.device.type != "cuda":
        raise ValueError("Fused kernels require CUDA tensors")
    return wp.from_torch(tensor, requires_grad=False)


def launch(kernel, dimensions, arguments, device):
    # Stay on the producer/consumer stream; never launch on an unrelated Warp stream.
    stream = wp.stream_from_torch(torch.cuda.current_stream(device=device))
    wp.launch(kernel, dim=dimensions, inputs=arguments, stream=stream)


class MotorKernels:
    def __init__(self, env):
        wp.init()
        self.env = env
        if env.dof_pos.dtype != torch.float32:
            raise ValueError("Warp servo kernels require float32; select --backend torch for float64")
        self.q = torch.empty_like(env.dof_pos)
        self.velocity = torch.empty_like(self.q)
        self.servo_arrays = [view(value) for value in (
            self.q, self.velocity, env.servo_target, env.servo_kp, env.servo_kd,
            env.servo_deadband, env.servo_backlash, env.servo_play_target,
            env.servo_torque, env.servo_speed,
            env.applied_torque, env.torque_sat, env.speed_sat,
            env.joint_torque_clip_count, env.joint_speed_clip_count, env.torque_effort)]

    def begin(self):
        # Own these snapshots: Genesis can update a zero-copy state view during a step.
        self.q.copy_(self.env.robot.get_dofs_position(self.env.motor_dofs))
        self.velocity.copy_(self.env.robot.get_dofs_velocity(self.env.motor_dofs))

    def servo(self):
        self.begin()
        launch(_servo_step, self.env.num_envs,
               [*self.servo_arrays, float(self.env.physics_steps), self.env.num_motors], self.env.device)


