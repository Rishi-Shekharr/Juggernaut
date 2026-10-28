"""Deterministic stage gates for saved checkpoints. Runs only when explicitly invoked."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    p.add_argument("--profile", choices=("nominal", "stage"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--seeds", type=int, nargs="+", default=[123, 456])
    p.add_argument("--commands", type=float, nargs="+", default=[0.10, 0.15])
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--baseline_speed", type=float)
    p.add_argument("--backend", choices=("auto", "torch", "warp"), default="auto")
    p.add_argument("--cpu", action="store_true")
    a = p.parse_args()
    if a.num_envs < 1 or not math.isfinite(a.seconds) or a.seconds < 5 or any(not math.isfinite(v) or v <= 0 for v in a.commands):
        p.error("Use positive environment counts/commands and at least 5 seconds")
    run = a.run.resolve()
    snapshot = run / "source"
    if not (snapshot / "juggernaut_env.py").is_file():
        p.error("A complete frozen V7 run is required")
    sys.path.insert(0, str(snapshot))
    import torch
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    from train import load_yaml, make_train_cfg, validate_checkpoint
    from juggernaut_env import JuggernautEnv

    saved = load_yaml(run / "config.yaml")
    joints = load_yaml(run / "joints.yaml")
    for checkpoint in a.checkpoints:
        validate_checkpoint(str(checkpoint.resolve()), saved, "Stage assessment", target_joints=joints)
    cfg = copy.deepcopy(saved)
    cfg["env"]["xml_path"] = str(snapshot / Path(cfg["env"]["xml_path"]).name)
    cfg["env"]["episode_length_s"] = a.seconds + 1.0
    cfg["env"]["collect_diagnostics"] = True
    cfg["performance"]["backend"] = a.backend
    cfg["commands"].pop("curriculum", None)
    if a.profile == "nominal":
        cfg["env"]["domain_randomization"]["enabled"] = False
        sensor = cfg["env"]["joint_sensor"]
        sensor.update(position_noise_std_rad=0.0, position_noise_uniform_rad=0.0, quantization_rad=0.0)
        cfg["env"]["observation_noise"] = {key: 0.0 for key in cfg["env"]["observation_noise"]}
    cuda = torch.cuda.is_available() and not a.cpu
    if not cuda and a.backend == "warp":
        p.error("Warp requires CUDA")
    torch.set_num_threads(int(cfg["performance"].get("torch_num_threads", 4)))
    gs.init(backend=gs.gpu if cuda else gs.cpu, seed=a.seeds[0], logging_level="error",
            performance_mode=cfg["performance"].get("genesis_performance_mode", True))
    env = JuggernautEnv(a.num_envs, cfg, joints, root_dir=str(snapshot))
    runner = OnPolicyRunner(env, make_train_cfg(cfg, "assessment", 0), None, device=gs.device)
    rows = []
    for checkpoint in a.checkpoints:
        runner.load(str(checkpoint.resolve()), load_optimizer=False)
        policy = runner.get_inference_policy(device=gs.device)
        trials = []
        for seed in a.seeds:
            for command in a.commands:
                torch.manual_seed(seed)
                if cuda:
                    torch.cuda.manual_seed_all(seed)
                env.set_fixed_command(command)
                with torch.inference_mode():
                    obs, _ = env.reset()
                    env.consume_training_metrics()
                alive = torch.ones(a.num_envs, device=gs.device, dtype=torch.bool)
                velocity_sum = torch.zeros(a.num_envs, device=gs.device)
                peaks = torch.zeros(2, device=gs.device)
                peak_gap_sum = torch.zeros((), device=gs.device)
                balance_sum = torch.zeros((), device=gs.device)
                swing_time_gap_sum = torch.zeros((), device=gs.device)
                swing_time_relative_gap_sum = torch.zeros((), device=gs.device)
                pair_count = torch.zeros((), device=gs.device)
                valid_pair_count = torch.zeros((), device=gs.device)
                obstruction = torch.zeros(a.num_envs, device=gs.device, dtype=torch.bool)
                measured_steps = 0
                with torch.inference_mode():
                    for step in range(math.ceil(a.seconds/env.dt)):
                        obs, _, done, extra = env.step(policy(obs))
                        d = extra["diagnostics"]
                        active = alive.clone()
                        obstruction |= active & d["arm_leg_contact"]
                        pair = active & d["completed_swing_pair"]
                        peaks += (d["completed_swing_peaks"] * pair[:, None]).sum(dim=0)
                        pair_peaks = d["completed_swing_peaks"]
                        pair_gap = (pair_peaks[:, 0] - pair_peaks[:, 1]).abs()
                        pair_balance = torch.where(
                            pair_peaks.amax(dim=1) > 1e-6,
                            pair_peaks.amin(dim=1) / pair_peaks.amax(dim=1).clamp_min(1e-6),
                            torch.zeros_like(pair_gap))
                        peak_gap_sum += (pair_gap * pair).sum()
                        balance_sum += (pair_balance * pair).sum()
                        swing_time_gap_sum += (d["completed_swing_time_gap"] * pair).sum()
                        swing_time_relative_gap_sum += (d["completed_swing_time_relative_gap"] * pair).sum()
                        pair_count += pair.sum()
                        valid_pair_count += (active & d["completed_swing_pair_valid"]).sum()
                        if (step+1)*env.dt > 2.0:
                            # Failed trajectories contribute zero thereafter; no survivor-only speed bias.
                            velocity_sum += d["base_lin_vel"][:, 0] * active
                            measured_steps += 1
                        alive &= ~done.bool()
                m = env.consume_training_metrics()
                mean_peaks = (peaks/pair_count.clamp_min(1)).tolist()
                mean_peak_gap = float(peak_gap_sum/pair_count.clamp_min(1)) if pair_count.item() else None
                mean_balance = float(balance_sum/pair_count.clamp_min(1)) if pair_count.item() else None
                mean_swing_time_gap = float(swing_time_gap_sum/pair_count.clamp_min(1)) if pair_count.item() else None
                mean_swing_time_gap_fraction = (float(swing_time_relative_gap_sum/pair_count.clamp_min(1))
                                                if pair_count.item() else None)
                valid_pair_ratio = float(valid_pair_count/pair_count.clamp_min(1)) if pair_count.item() else None
                completion = float(alive.float().mean())
                speed = float((velocity_sum/max(measured_steps, 1)).mean())
                budget = float(cfg["env"]["thermal_proxy"]["leg_rms_budget_nm"])
                rms = {key: value for key, value in m.items()
                       if key.startswith("motor/") and key.endswith("/rms_torque_nm")}
                # Runtime telemetry must expose all eight actual-substep leg RMS values.
                leg_names = [joint["name"] for joint in joints["joints"][:8]]
                leg_rms = [rms.get("motor/"+name+"/rms_torque_nm") for name in leg_names]
                thermal_ok = all(value is not None and value <= budget for value in leg_rms)
                checks = dict(completion=completion >= 0.95, speed=speed >= 0.85*command,
                              left_clearance=pair_count.item() > 0 and mean_peaks[0] >= 0.015,
                              right_clearance=pair_count.item() > 0 and mean_peaks[1] >= 0.015,
                              paired_peak_symmetry=mean_peak_gap is not None and mean_peak_gap <= 0.005,
                              paired_peak_balance=mean_balance is not None and mean_balance >= 0.75,
                              paired_time_symmetry=(mean_swing_time_gap_fraction is not None
                                                    and mean_swing_time_gap_fraction <= 0.25),
                              valid_pair_completion=valid_pair_ratio is not None and valid_pair_ratio >= 0.90,
                              arm_clearance=not bool(obstruction.any()), leg_rms_budget=thermal_ok)
                trials.append(dict(seed=seed, command_m_s=command, completion_fraction=completion,
                                   mean_speed_m_s=speed, completed_swing_peaks_m=mean_peaks,
                                   completed_pairs=int(pair_count), valid_pairs=int(valid_pair_count),
                                   valid_pair_completion_ratio=valid_pair_ratio,
                                   paired_swing_peak_gap_m=mean_peak_gap,
                                   paired_swing_balance_ratio=mean_balance,
                                   paired_swing_time_gap_s=mean_swing_time_gap,
                                   paired_swing_time_gap_fraction=mean_swing_time_gap_fraction,
                                   arm_obstruction_trajectories=int(obstruction.sum()),
                                   checks=checks, passed=all(checks.values()), telemetry=m))
        speed_mean = sum(t["mean_speed_m_s"] for t in trials)/len(trials)
        retention = a.baseline_speed is None or speed_mean >= 0.90*a.baseline_speed
        rows.append(dict(checkpoint=str(checkpoint.resolve()), profile=a.profile, trials=trials,
                         mean_speed_m_s=speed_mean, nominal_retention_passed=retention,
                         passed=retention and all(t["passed"] for t in trials)))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(profile=a.profile, results=rows), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print("Assessment report:", a.output, flush=True)
    return 0  # Gate failures are structured results; pipeline decides whether to advance.

if __name__ == "__main__":
    raise SystemExit(main())
