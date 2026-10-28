"""Final maneuver assessments. Failures are reported and never block completion."""
import argparse
import copy
import json
import math
import sys
from pathlib import Path

SCENARIOS = {
    "start_stop": [(0.0,0.0,4), (0.15,0.0,6), (0.0,0.0,6)],
    "slow_forward": [(0.05,0.0,8)],
    "forward_015": [(0.15,0.0,8)],
    "reverse": [(-0.08,0.0,8)],
    "turn_left": [(0.0,0.35,8)],
    "turn_right": [(0.0,-0.35,8)],
    "forward_left": [(0.12,0.30,8)],
    "forward_right": [(0.12,-0.30,8)],
    "reverse_turn": [(-0.08,0.25,8)],
}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--num_envs",type=int,default=16)
    p.add_argument("--backend",choices=("auto","torch","warp"),default="auto")
    p.add_argument("--seed",type=int,default=123)
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()
    if a.num_envs < 1: p.error("num_envs must be positive")
    run=a.run.resolve(); snapshot=run/"source"
    if not (snapshot/"juggernaut_env.py").is_file(): p.error("Run must contain its frozen runtime")
    sys.path.insert(0,str(snapshot))
    import torch
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    from train import load_yaml, make_train_cfg, validate_checkpoint
    from juggernaut_env import JuggernautEnv
    cfg=copy.deepcopy(load_yaml(run/"config.yaml")); joints=load_yaml(run/"joints.yaml")
    validate_checkpoint(str(a.checkpoint.resolve()),cfg,"Final assessment",target_joints=joints)
    cfg["env"]["xml_path"]=str(snapshot/Path(cfg["env"]["xml_path"]).name)
    cfg["env"]["episode_length_s"]=max(30.0,max(sum(s[2] for s in seq) for seq in SCENARIOS.values())+2)
    cfg["env"]["collect_diagnostics"]=True
    cfg["performance"]["backend"]=a.backend
    cuda=torch.cuda.is_available() and not a.cpu
    if a.backend=="warp" and not cuda: p.error("Warp requires CUDA")
    torch.set_num_threads(int(cfg["performance"].get("torch_num_threads",4)))
    gs.init(backend=gs.gpu if cuda else gs.cpu, seed=a.seed, logging_level="error",
            performance_mode=cfg["performance"].get("genesis_performance_mode",True))
    env=JuggernautEnv(a.num_envs,cfg,joints,root_dir=str(snapshot))
    runner=OnPolicyRunner(env,make_train_cfg(cfg,"assessment",0),None,device=gs.device)
    runner.load(str(a.checkpoint.resolve()),load_optimizer=False)
    policy=runner.get_inference_policy(device=gs.device)
    results=[]
    for name,sequence in SCENARIOS.items():
        torch.manual_seed(a.seed)
        if cuda: torch.cuda.manual_seed_all(a.seed)
        env.set_fixed_command(sequence[0][0],yaw=sequence[0][1])
        with torch.inference_mode():
            obs,_=env.reset(); env.consume_training_metrics()
            alive=torch.ones(a.num_envs,device=gs.device,dtype=torch.bool)
            obstructed=torch.zeros_like(alive)
            segment_results=[]
            for vx,yaw,seconds in sequence:
                env.set_fixed_command(vx,yaw=yaw)
                env._update_gait()
                env._compute_obs(advance_history=False)
                obs,_=env.get_observations()
                forward_sum=torch.zeros((),device=gs.device)
                yaw_sum=torch.zeros((),device=gs.device)
                sideways_sum=torch.zeros((),device=gs.device)
                samples=0
                steps=int(round(seconds/env.dt))
                for step in range(steps):
                    obs,_,done,extra=env.step(policy(obs))
                    d=extra["diagnostics"]; active=alive.clone()
                    obstructed |= active & d["arm_leg_contact"]
                    if step>=steps//2:
                        forward_sum+=(d["body_lin_vel"][:,0]*active).sum()
                        sideways_sum+=(d["body_lin_vel"][:,1].abs()*active).sum()
                        yaw_sum+=(d["body_yaw_rate"]*active).sum()
                        samples+=1
                    alive &= ~done.bool()
                den=max(samples*a.num_envs,1)
                measured_vx=float(forward_sum/den); measured_yaw=float(yaw_sum/den)
                checks={}
                if abs(vx)<1e-8:
                    checks["translation_stopped"]=abs(measured_vx)<=0.05
                else:
                    checks["forward_or_reverse"]=math.copysign(1.0,vx)*measured_vx >= (0.85 if vx==0.15 else 0.70)*abs(vx)
                if abs(yaw)<1e-8:
                    checks["yaw_stopped"]=abs(measured_yaw)<=0.18
                else:
                    checks["turn_direction_and_rate"]=math.copysign(1.0,yaw)*measured_yaw >= 0.5*abs(yaw)
                segment_results.append(dict(command_vx=vx,command_yaw=yaw,seconds=seconds,
                    mean_vx=measured_vx,mean_yaw_rate=measured_yaw,
                    mean_abs_lateral_speed=float(sideways_sum/den),checks=checks))
        telemetry=env.consume_training_metrics()
        budget=float(cfg["env"]["thermal_proxy"]["leg_rms_budget_nm"])
        leg_names=[j["name"] for j in joints["joints"][:8]]
        rms=[telemetry.get("motor/"+joint+"/rms_torque_nm") for joint in leg_names]
        checks={"completion":float(alive.float().mean())>=0.95,
                "arm_clearance":not bool(obstructed.any()),
                "leg_rms_budget":all(x is not None and x<=budget for x in rms)}
        passed=all(checks.values()) and all(all(s["checks"].values()) for s in segment_results)
        results.append(dict(name=name,passed=passed,checks=checks,completion_fraction=float(alive.float().mean()),
                            segments=segment_results,leg_rms_nm=dict(zip(leg_names,rms))))
        print(name, "PASS" if passed else "FAIL (soft)", flush=True)
    report=dict(checkpoint=str(a.checkpoint.resolve()), seed=a.seed, num_envs=a.num_envs,
                passed=all(r["passed"] for r in results),results=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print("Soft-check report:",a.output,flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
