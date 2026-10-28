"""Train one stage of the V7 walking-to-realism curriculum; never skips its predecessor."""
import argparse
import copy
import hashlib
import json
import os
import platform
import random
import re
import shutil
import time
from datetime import datetime
from importlib.metadata import version

import genesis as gs
import numpy as np
import torch
import yaml
from rsl_rl.runners import OnPolicyRunner
from juggernaut_env import (EXPECTED_JOINTS, LEG_JOINT_COUNT, V7_PIPELINE, JuggernautEnv,
                            validate_joint_config, validate_training_contract)

from stages import STAGES, CONFIG_FILES, DEFAULT_BUDGETS, REWARD_VERSION
from portability import path_reference

ROOT = os.path.dirname(os.path.abspath(__file__))
PIPELINE_VERSION = V7_PIPELINE
BOOTSTRAP_EXPERIMENT = "juggernaut_v7_bootstrap"
REALISTIC_EXPERIMENT = "juggernaut_v7_realistic"
ACTOR_OBS = 230
CRITIC_OBS = 53
POLICY_ACTION_JOINTS = tuple(name for name, _ in EXPECTED_JOINTS[:LEG_JOINT_COUNT])
EXTERNAL_ARM_JOINTS = tuple(name for name, _ in EXPECTED_JOINTS[LEG_JOINT_COUNT:])
SOURCE_FILES = tuple(path.name for path in __import__("pathlib").Path(ROOT).iterdir()
                     if path.is_file() and path.suffix in (".py", ".yaml", ".md", ".txt", ".json", ".xml"))



def atomic_json(path, data):
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, allow_nan=False)
    os.replace(path + ".tmp", path)


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def make_train_cfg(cfg, exp_name, max_iterations):
    ppo = copy.deepcopy(cfg["ppo"])
    algorithm = dict(ppo["algorithm"], class_name="PPO")
    policy = dict(ppo["policy"], class_name="ActorCritic")
    return {
        "algorithm": algorithm, "init_member_classes": {}, "policy": policy,
        "runner": {"algorithm_class_name": "PPO", "checkpoint": -1,
                   "experiment_name": exp_name, "load_run": -1, "log_interval": 1,
                   "max_iterations": max_iterations, "record_interval": -1,
                   "resume": False, "resume_path": None, "run_name": ""},
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": ppo["num_steps_per_env"], "save_interval": ppo["save_interval"],
        "empirical_normalization": None, "seed": ppo["seed"],
    }


def list_runs(log_root, exp_name):
    base = os.path.join(log_root, exp_name)
    if not os.path.isdir(base):
        return []
    return sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))


def resolve_run_dir(log_root, exp_name, run=None):
    if run:
        path = os.path.join(log_root, exp_name, run)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Run not found: {path}")
        return path
    runs = [name for name in list_runs(log_root, exp_name)
            if os.path.isfile(os.path.join(log_root, exp_name, name, "config.yaml"))
            and any(re.fullmatch(r"model_\d+\.pt", entry)
                    for entry in os.listdir(os.path.join(log_root, exp_name, name)))]
    if not runs:
        raise FileNotFoundError(f"No runs under {os.path.join(log_root, exp_name)}")
    return os.path.join(log_root, exp_name, runs[-1])


def resolve_checkpoint(run_dir, ckpt=-1):
    if ckpt >= 0:
        path = os.path.join(run_dir, f"model_{ckpt}.pt")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path
    found = [(int(m.group(1)), f) for f in os.listdir(run_dir)
             if (m := re.fullmatch(r"model_(\d+)\.pt", f))]
    if not found:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {run_dir}")
    return os.path.join(run_dir, max(found)[1])


def checkpoint_dimensions(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no model_state_dict: {path}")

    def layers(module):
        found = [(int(parts[1]), value) for key, value in state.items()
                 if len((parts := key.split("."))) == 3 and parts[0] == module
                 and parts[1].isdigit() and parts[2] == "weight"
                 and torch.is_tensor(value) and value.ndim == 2]
        return [v for _, v in sorted(found)]

    actor, critic, std = layers("actor"), layers("critic"), state.get("std")
    if not actor or not critic or not torch.is_tensor(std):
        raise ValueError(f"Checkpoint does not contain expected actor/critic policy weights: {path}")
    return (int(actor[0].shape[1]), int(actor[-1].shape[0]),
            int(critic[0].shape[1]), int(std.numel()))


def validate_checkpoint(path, cfg, context="Checkpoint", *, allow_stage_transition=False,
                        target_joints=None):
    run_dir = os.path.dirname(os.path.abspath(path))
    saved_cfg_path = os.path.join(run_dir, "config.yaml")
    saved_joints_path = os.path.join(run_dir, "joints.yaml")
    if not os.path.isfile(saved_cfg_path) or not os.path.isfile(saved_joints_path):
        raise ValueError(f"{context}: checkpoint needs its saved v7 config and joints: {run_dir}")
    saved = load_yaml(saved_cfg_path) or {}
    if (saved.get("training", {}).get("pipeline_version") != PIPELINE_VERSION
            or saved.get("training", {}).get("reward_version") != REWARD_VERSION):
        raise ValueError(f"{context}: checkpoint pipeline version is incompatible; expected {PIPELINE_VERSION}. Start a fresh v4 bootstrap run.")
    saved_policy = saved.get("ppo", {}).get("policy", {})
    target_policy = cfg.get("ppo", {}).get("policy", {})
    for key in ("actor_hidden_dims", "critic_hidden_dims", "activation"):
        if saved_policy.get(key) != target_policy.get(key):
            raise ValueError(f"{context}: checkpoint policy {key} differs from this run")
    source_stage = saved.get("training", {}).get("stage")
    target_stage = cfg.get("training", {}).get("stage")
    transition = (allow_stage_transition and source_stage in STAGES and target_stage in STAGES
                  and (source_stage == target_stage or STAGES.index(target_stage) == STAGES.index(source_stage)+1))
    if source_stage != target_stage and not transition:
        raise ValueError(f"{context}: cannot load {source_stage!r} into {target_stage!r}")
    expected = (ACTOR_OBS, LEG_JOINT_COUNT, CRITIC_OBS, LEG_JOINT_COUNT)
    if checkpoint_dimensions(path) != expected:
        raise ValueError(f"{context}: checkpoint must have 230 actor inputs, 53 critic inputs and 8 leg-action outputs")
    if saved.get("training_contract") != cfg.get("training_contract"):
        raise ValueError("Checkpoint observation contract differs from this run")
    saved_joints = load_yaml(saved_joints_path)
    validate_training_contract(saved, saved_joints)
    if target_joints is not None and saved_joints != target_joints:
        raise ValueError("Checkpoint joint order, gains, torque/speed caps, trims or action scales differ")
    for key in ("obs_scales", "gait"):
        if saved.get(key) != cfg.get(key):
            raise ValueError(f"Checkpoint {key} differs from this run")
    source_env, target_env = saved.get("env", {}), cfg.get("env", {})
    before_arms, after_arms = copy.deepcopy(source_env.get("external_arms")), copy.deepcopy(target_env.get("external_arms"))
    if transition:
        for candidate in (before_arms, after_arms):
            for name in ("mode", "curriculum"):
                candidate.pop(name, None)
    if before_arms != after_arms:
        raise ValueError(f"{context}: arm geometry, clearance envelope or command units differ")
    for key in ("control_dt", "substeps", "imu", "action_delay_steps",
                "command_reference_frame", "joint_observation", "foot_links", "base_link",
                "nominal_leg_actuator", "observation_history_frames", "init_base_pos"):
        if source_env.get(key) != target_env.get(key):
            raise ValueError(f"Checkpoint env.{key} differs from this run")
    for key in ("servo", "joint_sensor"):
        before, after = copy.deepcopy(source_env.get(key, {})), copy.deepcopy(target_env.get(key, {}))
        if transition:
            allowed = (("backlash_rad",) if key == "servo" else
                       ("position_noise_std_rad", "position_noise_uniform_rad", "quantization_rad"))
            for name in allowed:
                before.pop(name, None)
                after.pop(name, None)
        if before != after:
            raise ValueError(f"Checkpoint env.{key} differs from this run")
    if "foot_switch" in source_env or "foot_switch" in target_env:
        raise ValueError("The v7 hardware contract cannot contain foot-switch settings")
    for key in ("disable_state_projection", "joint_velocity_observation", "init_base_quat", "servo_safety",
                "self_collision", "adjacent_collision", "neutral_collision", "max_collision_pairs", "max_contacts"):
        if source_env.get(key) != target_env.get(key):
            raise ValueError(f"Checkpoint env.{key} differs from this run")
    return source_stage



def validate_actor_initialization(path, cfg, joints, xml_path):
    """Compatibility check for actor-only reuse; never accepts old value/optimizer state."""
    from pathlib import Path
    path = Path(path).resolve()
    run = path.parent
    old = load_yaml(run/"config.yaml")
    old_joints = load_yaml(run/"joints.yaml")
    if (old.get("training", {}).get("pipeline_version") != "juggernaut_v7_external_arms_history_v4"
            or old.get("training", {}).get("stage") != "realistic"):
        raise ValueError("Actor initialization requires a V7 realistic-stage history-policy checkpoint")
    if checkpoint_dimensions(str(path)) != (ACTOR_OBS, LEG_JOINT_COUNT, CRITIC_OBS, LEG_JOINT_COUNT):
        raise ValueError("Actor initialization dimensions are incompatible")
    old_contract = copy.deepcopy(old.get("training_contract"))
    if old_contract and old_contract.get("actor_fields", [])[2] == "commands_xy":
        old_contract["actor_fields"][2] = "commands_forward_yaw"
    if old_joints != joints or old_contract != cfg.get("training_contract"):
        raise ValueError("Actor initialization joint/observation contract differs beyond the repurposed command slot")
    for key in ("obs_scales", "gait"):
        if old.get(key) != cfg.get(key):
            raise ValueError("Actor initialization " + key + " differs")
    old_policy, new_policy = old["ppo"]["policy"], cfg["ppo"]["policy"]
    for key in ("actor_hidden_dims", "activation"):
        if old_policy.get(key) != new_policy.get(key):
            raise ValueError("Actor initialization architecture differs")
    for key in ("control_dt", "substeps", "imu", "joint_observation", "action_delay_steps",
                "nominal_leg_actuator", "servo", "init_base_pos", "init_base_quat", "base_link", "foot_links",
                "clip_actions", "observation_history_frames", "self_collision", "adjacent_collision", "neutral_collision"):
        if old["env"].get(key) != cfg["env"].get(key):
            raise ValueError("Actor initialization env." + key + " differs")
    archived_xml = run/"source"/Path(xml_path).name
    if not archived_xml.is_file() or sha256_file(str(archived_xml)) != sha256_file(xml_path):
        raise ValueError("Actor initialization robot XML differs or its snapshot is incomplete")
    for current in (Path(xml_path).parent/"meshes").rglob("*"):
        if current.is_file():
            archived = run/"source"/"meshes"/current.relative_to(Path(xml_path).parent/"meshes")
            if not archived.is_file() or sha256_file(str(current)) != sha256_file(str(archived)):
                raise ValueError("Actor initialization visual/collision mesh differs: " + current.name)


def initialize_actor(runner, path):
    """New critic, optimizer, noise and iteration schedule remain freshly initialized."""
    checkpoint = torch.load(path, map_location=runner.device, weights_only=False)
    actor = {key[len("actor."):]: value for key, value in checkpoint["model_state_dict"].items()
             if key.startswith("actor.")}
    runner.alg.actor_critic.actor.load_state_dict(actor, strict=True)
    # The old second command slot was always zero. Its weights are untrained;
    # zero them in all five history frames before exposing nonzero yaw commands.
    with torch.no_grad():
        runner.alg.actor_critic.actor[0].weight[:, [7 + 46*i for i in range(5)]] = 0.0
    runner.current_learning_iteration = 0


def apply_bootstrap(cfg):
    env = cfg["env"]
    env.setdefault("domain_randomization", {})["enabled"] = False
    env.setdefault("pushes", {})["enabled"] = False
    env.setdefault("servo", {})["backlash_rad"] = 0.0
    sensor = env.setdefault("joint_sensor", {})
    sensor["position_noise_std_rad"] = 0.0
    sensor["position_noise_uniform_rad"] = 0.0


class CompactRunner(OnPolicyRunner):
    def save(self, path, infos=None):
        infos = dict(infos or {})
        if hasattr(self.env, "get_training_curriculum_state"):
            infos["training_curriculum"] = self.env.get_training_curriculum_state()
        infos["learning_rate"] = float(self.alg.learning_rate)
        infos["completed_iterations"] = self.current_learning_iteration + 1
        temporary = os.fspath(path) + ".tmp"
        super().save(temporary, infos)
        os.replace(temporary, path)

    def load(self, path, load_optimizer=True):
        infos = super().load(path, load_optimizer=load_optimizer)
        if load_optimizer and isinstance(infos, dict):
            if "training_curriculum" in infos and hasattr(self.env, "load_training_curriculum_state"):
                self.env.load_training_curriculum_state(infos["training_curriculum"])
            if "learning_rate" in infos:
                self.alg.learning_rate = float(infos["learning_rate"])
                for group in self.alg.optimizer.param_groups:
                    group["lr"] = self.alg.learning_rate
        return infos

    def log(self, locs, *args, **kwargs):
        """Transfer episode scalars once per rollout, without duplicate base logging."""
        floor = float(self.env.cfg["ppo"].get("min_noise_std", 0.0))
        if floor > 0:
            with torch.no_grad():
                self.alg.actor_critic.std.clamp_(min=floor)
                ceiling = self.env.cfg["ppo"].get("max_noise_std")
                if ceiling is not None:
                    self.alg.actor_critic.std.clamp_(max=float(ceiling))
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        collection = float(locs["collection_time"])
        learning = float(locs["learn_time"])
        iteration_time = collection + learning
        self.tot_time += iteration_time
        iteration = int(locs["it"])
        episodes = locs.get("ep_infos", [])
        keys = list(episodes[0]) if episodes else []
        # The environment retains reward means on-device during collection.
        # Pack their reductions and policy noise into one host transfer here.
        reduced = [
            torch.stack([
                torch.as_tensor(episode[key], device=self.device).detach().float().mean()
                for episode in episodes if key in episode
            ]).mean()
            for key in keys
        ]
        reduced.append(self.alg.actor_critic.std.detach().mean())
        loss_names = [key for key in ("mean_value_loss", "mean_surrogate_loss", "mean_entropy",
                                      "mean_rnd_loss", "mean_symmetry_loss")
                      if locs.get(key) is not None]
        reduced.extend(torch.as_tensor(locs[key], device=self.device).detach().float().mean()
                       for key in loss_names)
        values = torch.stack(reduced).cpu().tolist()
        episode_values = dict(zip(keys, values[:len(keys)]))
        policy_std = values[len(keys)]
        losses = dict(zip(loss_names, values[len(keys)+1:]))
        rewards, lengths = list(locs["rewbuffer"]), list(locs["lenbuffer"])
        reward_mean = sum(rewards) / len(rewards) if rewards else None
        length_mean = sum(lengths) / len(lengths) if lengths else None
        training_metrics = self.env.consume_training_metrics()
        status = {
            "state": "running", "iteration": iteration,
            "target_iteration": int(locs["tot_iter"]) - 1,
            "updated_local": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": round(self.tot_time, 1),
            "iteration_seconds": round(iteration_time, 3),
            "mean_reward": round(reward_mean, 4) if reward_mean is not None else None,
            "mean_episode_seconds": round(length_mean * self.env.dt, 3) if length_mean is not None else None,
            "policy_std": round(policy_std, 4),
            "episode_metrics": {key: round(value, 5) for key, value in episode_values.items()},
            "optimization": {
                **{key: round(value, 6) for key, value in losses.items()
                   if key in ("mean_value_loss", "mean_surrogate_loss")},
                "collection_time": round(collection, 6), "learn_time": round(learning, 6),
            },
            "compute_backend": getattr(self.env, "compute_backend", None),
        }
        if hasattr(self.env, "get_training_curriculum_state"):
            status["curriculum"] = self.env.get_training_curriculum_state()
        status.update({key: None if value is None else round(value, 5)
                       for key, value in training_metrics.items()})
        atomic_json(os.path.join(self.log_dir, "status.json"), status)
        with open(os.path.join(self.log_dir, "history.jsonl"), "a", encoding="utf-8") as stream:
            stream.write(json.dumps(status, allow_nan=False) + "\n")
        start = int(self.env.cfg["training"].get("selection_start_iteration", 0))
        if iteration >= start and (iteration+1) % int(self.env.cfg["ppo"]["save_interval"]) == 0:
            speed = training_metrics.get("mean_forward_velocity_m_s")
            command = training_metrics.get("mean_command_forward_velocity_m_s")
            if speed is not None and command is not None and command > 0:
                ratio = speed / command
                duration = length_mean * self.env.dt if length_mean is not None else 0.0
                balance = float(training_metrics.get("paired_swing_balance_ratio") or 0.0)
                valid_pairs = float(training_metrics.get("valid_pair_completion_ratio") or 0.0)
                time_gap = training_metrics.get("paired_swing_time_gap_fraction")
                time_balance = max(0.0, 1.0-float(time_gap)) if time_gap is not None else 0.0
                left_peak = float(training_metrics.get("mean_left_completed_swing_peak_m") or 0.0)
                right_peak = float(training_metrics.get("mean_right_completed_swing_peak_m") or 0.0)
                lift_quality = min(1.0, min(left_peak, right_peak)/float(self.env.gait["swing_height"]))
                symmetry_quality = valid_pairs*balance*time_balance
                score = (3.0*symmetry_quality + 1.5*lift_quality + max(-1.0, min(1.0, ratio))
                         - max(0.0, 1.0-duration/float(self.env.env_cfg["episode_length_s"])))
                path = os.path.join(self.log_dir, f"model_{iteration}.pt")
                self.save(path)
                candidate_path = os.path.join(self.log_dir, "candidates.json")
                if os.path.isfile(candidate_path):
                    with open(candidate_path, encoding="utf-8") as stream:
                        candidates = json.load(stream)
                else:
                    candidates = []
                candidates = [item for item in candidates if os.path.basename(item["checkpoint"]) != os.path.basename(path)]
                candidates.append(dict(checkpoint=os.path.basename(path), score=score, iteration=iteration,
                                       symmetry_quality=symmetry_quality, weaker_foot_lift_quality=lift_quality,
                                       speed_ratio=ratio))
                candidates.sort(key=lambda item: item["score"], reverse=True)
                atomic_json(candidate_path, candidates[:3])
        if self.writer is not None:
            writer = self.writer
            for key, value in episode_values.items():
                writer.add_scalar(key if "/" in key else "Episode/" + key, value, iteration)
            for name, key in (("value_function", "mean_value_loss"),
                              ("surrogate", "mean_surrogate_loss"),
                              ("entropy", "mean_entropy")):
                if key in losses:
                    writer.add_scalar("Loss/" + name, losses[key], iteration)
            writer.add_scalar("Loss/learning_rate", self.alg.learning_rate, iteration)
            if self.alg.rnd and "mean_rnd_loss" in losses:
                writer.add_scalar("Loss/rnd", losses["mean_rnd_loss"], iteration)
            if self.alg.symmetry and "mean_symmetry_loss" in losses:
                writer.add_scalar("Loss/symmetry", losses["mean_symmetry_loss"], iteration)
            writer.add_scalar("Policy/mean_noise_std", policy_std, iteration)
            for key, value in status.get("curriculum", {}).items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    writer.add_scalar("Curriculum/" + key, value, iteration)
            writer.add_scalar("Perf/total_fps", int(self.num_steps_per_env * self.env.num_envs / max(iteration_time, 1e-9)), iteration)
            writer.add_scalar("Perf/collection time", collection, iteration)
            writer.add_scalar("Perf/learning_time", learning, iteration)
            if reward_mean is not None:
                writer.add_scalar("Train/mean_reward", reward_mean, iteration)
                writer.add_scalar("Train/mean_episode_length", length_mean, iteration)
                if getattr(self, "logger_type", "tensorboard") != "wandb":
                    writer.add_scalar("Train/mean_reward/time", reward_mean, self.tot_time)
                    writer.add_scalar("Train/mean_episode_length/time", length_mean, self.tot_time)
            for key, value in training_metrics.items():
                if value is not None:
                    writer.add_scalar("Train/" + key, value, iteration)
        if status["iteration"] % self.cfg.get("console_interval", 50) == 0:
            print(f"iter={status['iteration']}/{status['target_iteration']} reward={status['mean_reward']} "
                  f"episode_s={status['mean_episode_seconds']} "
                  f"vx={status.get('mean_forward_velocity_m_s')} "
                  f"cmd_vx={status.get('mean_command_forward_velocity_m_s')} "
                  f"vx_mae={status.get('forward_velocity_mae_m_s')} "
                  f"step_s={status['iteration_seconds']} backend={status['compute_backend']}", flush=True)
            print(f"  swing: left_peak={status.get('mean_left_completed_swing_peak_m')} "
                  f"right_peak={status.get('mean_right_completed_swing_peak_m')} "
                  f"gap={status.get('paired_swing_peak_gap_m')} "
                  f"target={status.get('target_peak_swing_clearance_m')} "
                  f"clear_l={status.get('mean_left_swing_clearance_m')} "
                  f"clear_r={status.get('mean_right_swing_clearance_m')}", flush=True)
            print(f"  symmetry: valid_pairs={status.get('valid_pair_completion_ratio')} "
                  f"peak_balance={status.get('paired_swing_balance_ratio')} "
                  f"airtime_gap_fraction={status.get('paired_swing_time_gap_fraction')} "
                  f"pair_samples={status.get('completed_pair_count')}", flush=True)
            print(f"  servo: slew_sat_pct={status.get('target_slew_saturation_pct')} "
                  f"torque_sat_pct={status.get('torque_saturation_pct')} "
                  f"speed_sat_pct={status.get('drive_speed_saturation_pct')} "
                  f"joint_overspeed_pct={status.get('joint_overspeed_pct')} "
                  f"hard_speed_limit_pct={status.get('hard_speed_limit_pct')} "
                  f"heading_error_abs_rad={status.get('heading_error_mean_abs_rad')} "
                  f"lateral_excursion_m={status.get('lateral_excursion_mean_abs_m')}", flush=True)
        # Public runner logging happens once after every completed PPO update.
        # Advance only the parameters sampled at subsequent episode resets.
        if hasattr(self.env, "set_training_iteration"):
            self.env.set_training_iteration(iteration + 1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--exp_name")
    p.add_argument("--num_envs", type=int)
    p.add_argument("--max_iterations", type=int, help="total stage budget, including completed iterations on resume")
    p.add_argument("--config")
    p.add_argument("--result_json", help="write run/checkpoint information for pipeline.py")
    p.add_argument("--joints")
    p.add_argument("--log_root", default=os.path.join(ROOT, "logs"))
    p.add_argument("--backend", choices=("auto", "warp", "torch"), default=None)
    p.add_argument("--cpu", action="store_true", help="run Genesis and policy on CPU")
    stage = p.add_mutually_exclusive_group()
    stage.add_argument("--stage", choices=STAGES)
    stage.add_argument("--bootstrap", dest="stage", action="store_const", const="bootstrap")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--load_checkpoint", help="explicit checkpoint from the immediately preceding stage")
    p.add_argument("--init_actor_checkpoint", help="optional compatible V7 v4 actor-only initialization for a NEW bootstrap; all other state resets")
    p.add_argument("--run")
    p.add_argument("--ckpt", type=int, default=-1)
    p.add_argument("--seed", type=int)
    p.add_argument("--delay", type=int)
    p.add_argument("--friction", type=float)
    p.add_argument("--console_interval", type=int)
    a = p.parse_args()

    if sum(bool(value) for value in (a.resume, a.load_checkpoint, a.init_actor_checkpoint)) > 1:
        p.error("Choose exact resume, stage warm start, or actor-only initialization")
    if a.run and not a.resume:
        p.error("--run selects a run to resume; use --load_checkpoint for a new stage")
    selected_stage = a.stage or "maneuver"
    if a.load_checkpoint and not a.stage and not a.config:
        p.error("Specify --stage when warm-starting; source stages are never guessed")
    default_experiment = "juggernaut_v7_" + selected_stage
    resume_dir = None
    if a.resume:
        try:
            resume_dir = resolve_run_dir(a.log_root, a.exp_name or default_experiment, a.run)
        except FileNotFoundError as error:
            p.error(str(error))
        resume_cfg = load_yaml(os.path.join(resume_dir, "config.yaml"))
        if (resume_cfg.get("training", {}).get("pipeline_version") != PIPELINE_VERSION
                or resume_cfg.get("training", {}).get("reward_version") != REWARD_VERSION):
            p.error("This checkpoint uses an incompatible pipeline/action mapping. Start a fresh v4 pipeline; v1/v2/v3 checkpoints cannot resume or initialize the actor.")
        archived_train = os.path.join(resume_dir, "source", "train.py")
        if not os.path.isfile(archived_train):
            p.error("Resume requires its complete archived runtime")
        if os.path.realpath(__file__) != os.path.realpath(archived_train):
            argv = list(__import__("sys").argv[1:])
            if "--log_root" not in argv:
                argv += ["--log_root", os.path.abspath(a.log_root)]
            os.execv(__import__("sys").executable, [__import__("sys").executable, archived_train, *argv])
        config_path = a.config or os.path.join(resume_dir, "config.yaml")
        joints_path = a.joints or os.path.join(resume_dir, "joints.yaml")
    else:
        config_path = a.config or os.path.join(ROOT, CONFIG_FILES[selected_stage])
        joints_path = a.joints or os.path.join(ROOT, "joints.yaml")
    try:
        cfg, joints = load_yaml(config_path), load_yaml(joints_path)
    except OSError as error:
        p.error(str(error))
    training = cfg.setdefault("training", {})
    selected_stage = training.get("stage")
    if selected_stage not in STAGES or (a.stage and a.stage != selected_stage):
        p.error("--stage and training.stage must agree with a V7 stage")
    if selected_stage == "bootstrap" and (a.load_checkpoint):
        p.error("bootstrap must start fresh; use --resume to continue it")
    if a.init_actor_checkpoint and selected_stage != "maneuver":
        p.error("Actor initialization is supported only for the maneuver stage")
    if training.get("reward_version") != REWARD_VERSION:
        p.error("config must set training.reward_version: " + REWARD_VERSION)
    if training.get("pipeline_version") != PIPELINE_VERSION:
        p.error(f"config must set training.pipeline_version: {PIPELINE_VERSION}")
    exp_name = a.exp_name or training.get("experiment_name") or default_experiment
    default_budget = DEFAULT_BUDGETS[selected_stage]
    for name, fallback in (("num_envs", 2048), ("max_iterations", default_budget), ("console_interval", 1)):
        value = getattr(a, name)
        if value is None:
            value = training.get(name, fallback)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            p.error(f"{name} must be a positive integer")
        setattr(a, name, value)
    if not a.resume:
        training["num_envs"] = a.num_envs
        training["max_iterations"] = a.max_iterations
        training["console_interval"] = a.console_interval
    seed = a.seed if a.seed is not None else cfg["ppo"]["seed"]
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        p.error("seed must be an integer in [0, 2**32)")
    cfg["ppo"]["seed"] = seed
    if a.delay is not None:
        if a.delay < 0:
            p.error("delay must be nonnegative")
        cfg["env"]["action_delay_steps"] = a.delay
        cfg["env"].setdefault("domain_randomization", {})["action_delay_steps"] = [a.delay, a.delay]
    if a.friction is not None:
        cfg["env"]["friction"] = None if a.friction < 0 else {"floor": a.friction, "foot": a.friction}


    if not a.resume and selected_stage == "bootstrap":
        apply_bootstrap(cfg)
    try:
        validate_joint_config(joints)
        validate_training_contract(cfg, joints)
    except ValueError as error:
        p.error(str(error))

    xml_path = os.path.abspath(os.path.join(ROOT, cfg["env"]["xml_path"]))
    if not os.path.isfile(xml_path):
        p.error(f"robot XML not found: {xml_path}")
    if version("rsl-rl-lib") != "2.2.4":
        p.error("This runner requires rsl-rl-lib==2.2.4; use requirements-training.txt")

    device_has_cuda = torch.cuda.is_available() and not a.cpu
    requested_backend = a.backend or cfg.get("performance", {}).get("backend", "auto")
    if a.cpu and requested_backend == "warp":
        p.error("--cpu cannot be combined with --backend warp")
    backend = requested_backend
    if backend == "auto":
        backend = "warp" if device_has_cuda else "torch"
    if backend == "warp":
        if not device_has_cuda:
            p.error("Warp backend requires CUDA; remove --cpu or select --backend torch")
        try:
            import warp  # noqa: F401
        except ImportError as error:
            p.error(f"Warp backend requires warp-lang; install requirements-training.txt ({error})")
    gs_backend = gs.gpu if device_has_cuda else gs.cpu
    cfg.setdefault("performance", {})["backend"] = backend

    if a.resume:
        run_dir = resolve_run_dir(a.log_root, exp_name, a.run)
        saved_cfg = load_yaml(os.path.join(run_dir, "config.yaml"))
        saved_joints = load_yaml(os.path.join(run_dir, "joints.yaml"))
        compared_cfg, compared_saved_cfg = copy.deepcopy(cfg), copy.deepcopy(saved_cfg)
        for candidate in (compared_cfg, compared_saved_cfg):
            candidate_training = candidate.get("training", {})
            for key in ("max_iterations", "num_envs", "console_interval"):
                candidate_training.pop(key, None)
        # The selected runtime backend is an operational override and may change on resume.
        compared_cfg.setdefault("performance", {}).pop("backend", None)
        compared_saved_cfg.setdefault("performance", {}).pop("backend", None)
        for candidate in (compared_cfg, compared_saved_cfg):
            candidate.setdefault("performance", {}).pop("defer_inner_projection_forward", None)
        if compared_saved_cfg != compared_cfg or saved_joints != joints:
            p.error(f"resume config or joints differ from the saved run; expected {PIPELINE_VERSION} configuration")
        checkpoint = resolve_checkpoint(run_dir, a.ckpt)
        try:
            validate_checkpoint(checkpoint, cfg, "Resume", target_joints=joints)
        except (ValueError, OSError, RuntimeError) as error:
            p.error(str(error))
    else:
        run_dir = None
        checkpoint = None
        if selected_stage != "bootstrap" and not a.init_actor_checkpoint:
            try:
                if not a.load_checkpoint:
                    raise ValueError("Every new stage after bootstrap requires --load_checkpoint from its predecessor")
                checkpoint = os.path.abspath(a.load_checkpoint)
                validate_checkpoint(checkpoint, cfg, "Stage warm start",
                                    allow_stage_transition=True, target_joints=joints)
                source_xml = os.path.join(os.path.dirname(checkpoint), "source", os.path.basename(xml_path))
                if not os.path.isfile(source_xml) or sha256_file(source_xml) != sha256_file(xml_path):
                    raise ValueError("Warm-start robot XML differs or its archived source XML is missing")
                for directory, _, files in os.walk(os.path.join(ROOT, "meshes")):
                    for name in files:
                        current = os.path.join(directory, name)
                        relative = os.path.relpath(current, ROOT)
                        archived = os.path.join(os.path.dirname(checkpoint), "source", relative)
                        if not os.path.isfile(archived) or sha256_file(current) != sha256_file(archived):
                            raise ValueError("Warm-start collision/visual mesh differs: " + relative)
            except (ValueError, OSError, RuntimeError) as error:
                p.error(str(error))

    actor_initialization = None
    if a.init_actor_checkpoint:
        actor_initialization = os.path.abspath(a.init_actor_checkpoint)
        try:
            validate_actor_initialization(actor_initialization, cfg, joints, xml_path)
        except (ValueError, OSError, RuntimeError) as error:
            p.error(str(error))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(int(cfg.get("performance", {}).get("torch_num_threads", 4)))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("performance", {}).get("allow_tf32", True))
    torch.backends.cudnn.allow_tf32 = bool(cfg.get("performance", {}).get("allow_tf32", True))
    gs.init(backend=gs_backend, logging_level="error", seed=seed,
            performance_mode=bool(cfg.get("performance", {}).get("genesis_performance_mode", True)))

    if not a.resume:
        run_dir = os.path.join(a.log_root, exp_name, datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        os.makedirs(run_dir)
        save_yaml(cfg, os.path.join(run_dir, "config.yaml"))
        save_yaml(joints, os.path.join(run_dir, "joints.yaml"))
        snapshot = os.path.join(run_dir, "source")
        os.makedirs(snapshot)
        for name in SOURCE_FILES:
            source_path = os.path.join(ROOT, name)
            if os.path.isfile(source_path):
                shutil.copy2(source_path, os.path.join(snapshot, name))
        shutil.copy2(xml_path, os.path.join(snapshot, os.path.basename(xml_path)))
        meshes = os.path.join(os.path.dirname(xml_path), "meshes")
        if os.path.isdir(meshes):
            shutil.copytree(meshes, os.path.join(snapshot, "meshes"))
        diagnostics = os.path.join(ROOT, "diagnostics")
        if os.path.isdir(diagnostics):
            shutil.copytree(diagnostics, os.path.join(snapshot, "diagnostics"))
        warm_start = None
        if checkpoint is not None:
            frozen_checkpoint = os.path.join(run_dir, "warm_start.pt")
            shutil.copy2(checkpoint, frozen_checkpoint)
            warm_start = {"checkpoint": path_reference(checkpoint, run_dir), "frozen_checkpoint": "warm_start.pt",
                          "sha256": sha256_file(frozen_checkpoint)}
            checkpoint = frozen_checkpoint
        actor_init_info = None
        if actor_initialization is not None:
            frozen_actor = os.path.join(run_dir, "actor_init.pt")
            shutil.copy2(actor_initialization, frozen_actor)
            actor_init_info = {"source_checkpoint": path_reference(actor_initialization, run_dir),
                               "frozen_checkpoint": "actor_init.pt",
                               "sha256": sha256_file(frozen_actor),
                               "reused": "actor weights only; new critic, optimizer, noise and curriculum"}
            actor_initialization = frozen_actor
        source_hashes = {}
        for directory, _, filenames in os.walk(snapshot):
            for filename in sorted(filenames):
                path = os.path.join(directory, filename)
                source_hashes[os.path.relpath(path, snapshot).replace(os.sep, "/")] = sha256_file(path)
        atomic_json(os.path.join(run_dir, "metadata.json"), {
            "created_local": datetime.now().isoformat(), "seed": seed,
            "num_envs": a.num_envs, "stage": selected_stage, "max_iterations": a.max_iterations,
            "actor_initialization": actor_init_info,
            "reward_version": REWARD_VERSION,
            "warm_start": warm_start, "curriculum": copy.deepcopy(cfg.get("curriculum", training.get("curriculum"))),
            "pipeline_version": PIPELINE_VERSION, "actor_obs": ACTOR_OBS, "critic_obs": CRITIC_OBS,
            "num_actions": LEG_JOINT_COUNT, "policy_action_joints": list(POLICY_ACTION_JOINTS),
            "external_arm_joints": list(EXTERNAL_ARM_JOINTS),
            "actor_observation_contract": cfg["training_contract"]["actor_fields"],
            "critic_observation_contract": cfg["training_contract"]["critic_fields"],
            "model_spec": {"xml": os.path.basename(xml_path), "derived_by": "training_model.prepare_training_mjcf",
                           "floor_source": "environment-generated floor; XML floor omitted"},
            "python": platform.python_version(),
            "packages": {n: version(n) for n in
                         ("torch", "torchvision", "numpy", "PyYAML", "tensorboard",
                          "genesis-world", "rsl-rl-lib", "warp-lang")},
            "backend_requested": requested_backend, "backend_selected": backend,
            "gpu": torch.cuda.get_device_name(0) if device_has_cuda else "CPU",
            "xml_sha256": sha256_file(xml_path), "source_sha256": source_hashes,
        })
    def write_result(state, checkpoint_path=None, completed_iterations=0):
        if a.result_json:
            result_path = os.path.abspath(a.result_json)
            os.makedirs(os.path.dirname(result_path), exist_ok=True)
            atomic_json(result_path, {
                "state": state, "stage": selected_stage, "path_base": "result_directory",
                "run_dir": path_reference(run_dir, os.path.dirname(result_path)),
                "checkpoint": path_reference(checkpoint_path, os.path.dirname(result_path)) if checkpoint_path else None,
                "completed_iterations": int(completed_iterations),
                "target_iterations": a.max_iterations,
            })

    write_result("initialized")
    print(f"Stage: {selected_stage}; environments: {a.num_envs}; iterations: {a.max_iterations}; backend: {backend}", flush=True)
    print(f"Run directory: {run_dir}", flush=True)
    env = JuggernautEnv(a.num_envs, cfg, joints, show_viewer=False, root_dir=ROOT)
    print(f"[performance] device={gs.device}; "
          "post_physics_projection=disabled; "
          "GPU-batched episode statistics; per-stage actuator setup", flush=True)
    train_cfg = make_train_cfg(cfg, exp_name, a.max_iterations)
    train_cfg["console_interval"] = a.console_interval
    runner = CompactRunner(env, train_cfg, run_dir, device=gs.device)
    if a.resume:
        loaded_infos = runner.load(checkpoint)
        if not isinstance(loaded_infos, dict) or "completed_iterations" not in loaded_infos:
            p.error("Resume checkpoint is missing completed iteration metadata")
        runner.current_learning_iteration = int(loaded_infos["completed_iterations"])
    elif actor_initialization is not None:
        initialize_actor(runner, actor_initialization)
        print("Initialized actor only; critic, optimizer, noise and curricula are fresh", flush=True)
    elif checkpoint is not None:
        runner.load(checkpoint, load_optimizer=False)
        runner.current_learning_iteration = 0
        with torch.no_grad():
            runner.alg.actor_critic.std.clamp_(
                min=float(cfg["ppo"].get("min_noise_std", 0.0)),
                max=float(cfg["ppo"].get("max_noise_std", float("inf"))))
    # Initialize arm-pose progression in bootstrap as well as realistic dynamics.
    env.set_training_iteration(runner.current_learning_iteration)
    env.reset()
    iterations = a.max_iterations - runner.current_learning_iteration
    if iterations <= 0:
        write_result("complete", checkpoint, runner.current_learning_iteration)
        print(f"Stage budget of {a.max_iterations} iterations already complete", flush=True)
        return
    started = time.time()
    try:
        runner.learn(num_learning_iterations=iterations, init_at_random_ep_len=False)
    except KeyboardInterrupt:
        runner.save(os.path.join(run_dir, f"model_{runner.current_learning_iteration}.pt"))
        atomic_json(os.path.join(run_dir, "interrupted.json"), {"iteration": runner.current_learning_iteration})
        raise
    finally:
        if runner.writer:
            runner.writer.flush()
            runner.writer.close()
    final_checkpoint = os.path.join(run_dir, f"model_{runner.current_learning_iteration}.pt")
    runner.save(final_checkpoint)
    atomic_json(os.path.join(run_dir, "completed.json"), {
        "iteration": runner.current_learning_iteration, "elapsed_seconds": round(time.time() - started, 1),
        "checkpoint": os.path.basename(final_checkpoint),
    })
    status_path = os.path.join(run_dir, "status.json")
    if os.path.isfile(status_path):
        with open(status_path, encoding="utf-8") as f:
            status = json.load(f)
    else:
        status = {"iteration": runner.current_learning_iteration, "compute_backend": getattr(env, "compute_backend", backend)}
    status["state"] = "stage_finished"
    atomic_json(status_path, status)
    write_result("complete", final_checkpoint, runner.current_learning_iteration + 1)
    print(f"Finished: {run_dir} / {os.path.basename(final_checkpoint)}", flush=True)


if __name__ == "__main__":
    main()





