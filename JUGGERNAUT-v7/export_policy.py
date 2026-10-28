"""Export a frozen V7 actor, bounded actions and its deployment contract; does not run inference."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    checkpoint = a.checkpoint.resolve()
    run = checkpoint.parent
    snapshot = run/"source"
    if not (snapshot/"deployment.py").is_file():
        p.error("Checkpoint must have a complete V7 source snapshot")
    sys.path.insert(0, str(snapshot))
    import numpy as np
    import torch
    from train import load_yaml, validate_checkpoint
    cfg = load_yaml(run/"config.yaml")
    from stages import PIPELINE_VERSION, REWARD_VERSION
    if (cfg["training"].get("pipeline_version") != PIPELINE_VERSION
            or cfg["training"].get("reward_version") != REWARD_VERSION):
        p.error("Export requires a V7 v4 checkpoint with the current reward and action contracts; use its archived exporter for older runs")
    joints = load_yaml(run/"joints.yaml")
    validate_checkpoint(str(checkpoint), cfg, "Deployment export")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model_state_dict"]
    sizes = [230, *cfg["ppo"]["policy"]["actor_hidden_dims"], 8]
    activations = {"elu": torch.nn.ELU, "relu": torch.nn.ReLU, "tanh": torch.nn.Tanh}
    activation = activations[cfg["ppo"]["policy"]["activation"].lower()]
    layers = []
    arrays = {}
    for i, (input_size, output_size) in enumerate(zip(sizes[:-1], sizes[1:])):
        layer = torch.nn.Linear(input_size, output_size)
        source_index = 2*i
        with torch.no_grad():
            layer.weight.copy_(state[f"actor.{source_index}.weight"])
            layer.bias.copy_(state[f"actor.{source_index}.bias"])
        arrays[f"layer_{i}_weight"] = layer.weight.detach().numpy()
        arrays[f"layer_{i}_bias"] = layer.bias.detach().numpy()
        layers.append(layer)
        if i < len(sizes)-2:
            layers.append(activation())
    actor = torch.nn.Sequential(*layers, torch.nn.Hardtanh(-1.0, 1.0)).eval()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    # Scripting exports the graph. No inference, tracing rollout, or simulation.
    torch.jit.script(actor).save(str(output/"actor.pt"))
    np.savez(output/"actor_weights.npz", **arrays)
    shutil.copy2(snapshot/"deployment.py", output/"deployment.py")
    shutil.copy2(snapshot/"external_arms.py", output/"external_arms.py")
    shutil.copy2(snapshot/"sensors.py", output/"sensors.py")
    motors = [dict(joints["defaults"], **joint) for joint in joints["joints"]]
    for motor in motors:
        motor.setdefault("target_limit", joints["target_limit"])
    contract = {
         "pipeline_version":cfg["training"]["pipeline_version"],
        "reward_version":cfg["training"]["reward_version"],
        "source_checkpoint":str(checkpoint),
        "checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_stage":cfg["training"]["stage"],
        "gate_acceptance":"Consult pipeline.json and its reports; export alone does not approve a checkpoint.",
        "control_dt_s":cfg["env"]["control_dt"],
        "observation_contract":cfg["training_contract"],
        "frame_slices":{
            "imu_body_gyro_scaled":[0,3],"imu_projected_gravity":[3,6],
            "scaled_commands_xy":[6,8],"scaled_leg_pot_positions_then_command_estimated_arms":[8,20],
            "scaled_nominal_command_observer_velocities":[20,32],"previous_clipped_leg_actions":[32,40],
            "scaled_external_arm_commands":[40,44],"gait_sin_cos":[44,46]},
        "observation_scales":cfg["obs_scales"],
        "imu":cfg["env"]["imu"],
        "joint_sensor":cfg["env"]["joint_sensor"],
        "nominal_observer_delay_steps":cfg["env"]["action_delay_steps"],
        "nominal_leg_actuator_model":cfg["env"]["nominal_leg_actuator"],
        "servo_target_processing":cfg["env"]["servo"],
        "gait":cfg["gait"],
        "motors":motors,
        "external_arms":cfg["env"]["external_arms"],
        "action_clip":[-1.0,1.0],
        "leg_target_formula":"clamp(default_pos + pd_trim + clipped_action * action_scale, target_limit)",
        "arm_policy_outputs":0,
        "notes":[
            "Eight leg potentiometers and a six-axis IMU are required by this actor.",
            "Arm positions and all joint-velocity features are nominal software estimates from issued targets; no actual arm state, force, heading or simulator velocity enters the actor.",
            "Repeat the initial frame five times, append exactly once each 20 ms; use calibrated hardware signs and zero offsets before applying model angles.",
            "Direct/Bluetooth arm requests must use the exported raised envelope and command speed, never bypass the guard.",
            "MG996R PWM accepts position targets; simulated gain/torque parameters model its response and are not configurable hardware torque commands.",
            "The 4 kgf cm design RMS budget is provisional. Calibrate response, latency, voltage sag and heating on the actual robot before autonomous walking."
        ]}
    (output/"deployment.json").write_text(json.dumps(contract,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Exported:",output)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
