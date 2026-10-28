"""Preview or record an existing JUGGERNAUT v7 bootstrap or realistic checkpoint; never starts training.

python eval.py --stage bootstrap --camera rear
python eval.py --stage bootstrap --no_viewer --record walking.mp4 --seconds 15
"""
from stages import STAGES

import argparse
import math
import os
import sys
from pathlib import Path


def resolve_selected_run(log_root, experiment, run=None):
    base = Path(log_root) / experiment
    if run:
        selected = base / run
        if not selected.is_dir():
            raise FileNotFoundError(f"Run not found: {selected}")
        return selected.resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"No runs under {base}")
    runs = sorted(path for path in base.iterdir() if path.is_dir())
    if not runs:
        raise FileNotFoundError(f"No runs under {base}")
    return runs[-1].resolve()


def archived_runtime(run_dir):
    snapshot = run_dir / "source"
    if not snapshot.exists():
        return None
    required = ("train.py", "juggernaut_env.py", "sensors.py", "cuda_kernels.py", "training_model.py", "external_arms.py", "stages.py", "deployment.py")
    missing = [name for name in required if not (snapshot / name).is_file()]
    if not ((snapshot / "viewer_scene.py").is_file() or (snapshot / "viewer.py").is_file()):
        missing.append("viewer_scene.py (or viewer.py)")
    if missing:
        raise ValueError(f"Run snapshot is incomplete at {snapshot}; missing runtime files: {', '.join(missing)}")
    return snapshot.resolve()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=STAGES, default="bootstrap")
    p.add_argument("--exp_name")
    p.add_argument("--log_root", default=str(Path(__file__).resolve().parent/"logs"))
    p.add_argument("--run")
    p.add_argument("--ckpt", type=int, default=-1)
    p.add_argument("--cmd", type=float, default=0.15)
    p.add_argument("--arms_deg", type=float, nargs=4, help="L shoulder, L mandible, R shoulder, R mandible; safe envelope enforced")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--camera", choices=("three_quarter", "side", "top", "rear"), default="rear")
    p.add_argument("--no_viewer", action="store_true")
    p.add_argument("--record", help="MP4 output path")
    p.add_argument("--record_start", type=float, default=0.0, help="seconds to wait before recording")
    p.add_argument("--record_seconds", type=float, default=None, help="recording duration within --seconds")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--backend", choices=("auto", "warp", "torch"))
    p.add_argument("--seed", type=int, default=123)
    a = p.parse_args()
    if not math.isfinite(a.seconds) or a.seconds <= 0 or not math.isfinite(a.cmd):
        p.error("seconds must be positive and command finite")
    if not math.isfinite(a.record_start) or a.record_start < 0:
        p.error("record_start must be finite and nonnegative")
    if a.record_seconds is not None and (not math.isfinite(a.record_seconds) or a.record_seconds <= 0):
        p.error("record_seconds must be finite and positive")
    if a.record and a.record_start >= a.seconds:
        p.error("record_start must be before the preview ends")
    experiment = a.exp_name or ("juggernaut_v7_" + a.stage)
    try:
        run = resolve_selected_run(a.log_root, experiment, a.run)
        runtime = archived_runtime(run)
    except (FileNotFoundError, ValueError) as error:
        p.error(str(error))
    if runtime is not None:
        sys.path.insert(0, str(runtime))
    import torch
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    from train import (load_yaml, make_train_cfg, resolve_checkpoint, validate_checkpoint)
    from juggernaut_env import JuggernautEnv, validate_training_contract
    try:
        checkpoint = resolve_checkpoint(run, a.ckpt)
        cfg = load_yaml(os.path.join(run, "config.yaml"))
        joints = load_yaml(os.path.join(run, "joints.yaml"))
        validate_checkpoint(checkpoint, cfg, "Preview")
        validate_training_contract(cfg, joints)
    except (ValueError, OSError, RuntimeError) as error:
        p.error(str(error))
    performance = cfg.setdefault("performance", {})
    performance["backend"] = a.backend or performance.get("backend", "auto")
    cuda = torch.cuda.is_available() and not a.cpu
    if not cuda and performance["backend"] == "warp":
        if a.backend == "warp":
            p.error("Warp requires CUDA")
        performance["backend"] = "torch"
    cfg.setdefault("viewer", {})["camera_view"] = a.camera
    cfg["env"]["episode_length_s"] = max(float(cfg["env"]["episode_length_s"]), a.seconds+1)
    torch.set_num_threads(int(performance.get("torch_num_threads", 4)))
    gs.init(backend=gs.gpu if cuda else gs.cpu, seed=a.seed, logging_level="warning",
            performance_mode=bool(performance.get("genesis_performance_mode", True)))
    # Prefer the archived XML/meshes of this checkpoint; source tree is fallback.
    archive = Path(run)/"source"
    xml = archive/Path(cfg["env"]["xml_path"]).name
    if xml.is_file():
        cfg["env"]["xml_path"] = str(xml.resolve())
    root = str(Path(__file__).resolve().parent)
    env = JuggernautEnv(1, cfg, joints, show_viewer=not a.no_viewer,
                        record_camera=bool(a.record), root_dir=root)
    runner = OnPolicyRunner(env, make_train_cfg(cfg, experiment, 0), None, device=gs.device)
    runner.load(checkpoint, load_optimizer=False)
    policy = runner.get_inference_policy(device=gs.device)
    env.set_fixed_command(a.cmd)
    obs, _ = env.reset()
    if a.arms_deg is not None:
        env.set_external_arm_commands([math.radians(v) for v in a.arms_deg])
        obs, _ = env.get_observations()
    recording = False
    destination = str(Path(a.record).resolve()) if a.record else None
    if destination:
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
    record_start_step = math.ceil(a.record_start/env.dt-1e-9)
    record_end_time = a.seconds if a.record_seconds is None else min(a.seconds, a.record_start+a.record_seconds)
    record_end_step = math.floor(record_end_time/env.dt+1e-9)
    if destination and record_end_step <= record_start_step:
        p.error("recording window must include at least one complete 20 ms control interval")
    falls = 0
    try:
        with torch.inference_mode():
            for step in range(math.ceil(a.seconds/env.dt)):
                if destination and record_start_step <= step < record_end_step and not recording:
                    env.cam.start_recording(save_to_filename=destination, fps=round(1/env.dt))
                    recording = True
                obs, _, done, _ = env.step(policy(obs))
                falls += int(env.fell_buf.sum().item())
                if env.presentation is not None:
                    env.presentation.update()
                # Genesis 1.4 streams camera frames inside scene.step(), as in
                # 4dof/eval.py; manual render/readback is unnecessary for video.
                if recording and step+1 >= record_end_step:
                    env.cam.stop_recording()
                    recording = False
    finally:
        if recording:
            env.cam.stop_recording()
    if destination:
        print("Video:", destination)
    metrics = env.consume_training_metrics()
    print("Checkpoint:", checkpoint)
    print("Falls:", falls)
    print("Completed swing peaks L/R (m):", metrics["mean_left_completed_swing_peak_m"], metrics["mean_right_completed_swing_peak_m"])
    print("Mean completed pair gap (m):", metrics["paired_swing_peak_gap_m"])


if __name__ == "__main__":
    main()
