"""Drive a trained maneuver checkpoint in the Genesis viewer with a joystick."""
import argparse
import sys
import time
from pathlib import Path
from joystick import command_from_axes

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run",required=True,help="maneuver run folder name under logs/juggernaut_v7_maneuver")
    p.add_argument("--ckpt",type=int,default=-1)
    p.add_argument("--axis_x",type=int,default=0)
    p.add_argument("--axis_y",type=int,default=1)
    p.add_argument("--deadzone",type=float,default=0.08)
    p.add_argument("--backend",choices=("auto","torch","warp"),default="auto")
    p.add_argument("--cpu",action="store_true")
    a=p.parse_args()
    root=Path(__file__).resolve().parent
    run=root/"logs"/"juggernaut_v7_maneuver"/a.run
    snapshot=run/"source"
    if not snapshot.is_dir(): p.error("Run with frozen source not found")
    sys.path.insert(0,str(snapshot))
    import pygame
    import torch
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    from train import load_yaml,make_train_cfg,resolve_checkpoint,validate_checkpoint
    from juggernaut_env import JuggernautEnv
    cfg=load_yaml(run/"config.yaml"); joints=load_yaml(run/"joints.yaml")
    checkpoint=resolve_checkpoint(str(run),a.ckpt)
    validate_checkpoint(checkpoint,cfg,"Joystick preview",target_joints=joints)
    pygame.init(); pygame.joystick.init()
    if pygame.joystick.get_count()==0: p.error("Connect a joystick before starting")
    stick=pygame.joystick.Joystick(0); stick.init()
    if min(a.axis_x,a.axis_y)<0 or max(a.axis_x,a.axis_y)>=stick.get_numaxes(): p.error("Joystick axis index is out of range")
    command_from_axes(0,0,a.deadzone)
    cuda=torch.cuda.is_available() and not a.cpu
    if a.backend=="warp" and not cuda: p.error("Warp requires CUDA")
    cfg["performance"]["backend"]=a.backend
    cfg["env"]["xml_path"]=str(snapshot/Path(cfg["env"]["xml_path"]).name)
    gs.init(backend=gs.gpu if cuda else gs.cpu,seed=123,logging_level="warning",
            performance_mode=cfg["performance"].get("genesis_performance_mode",True))
    env=JuggernautEnv(1,cfg,joints,show_viewer=True,root_dir=str(snapshot))
    runner=OnPolicyRunner(env,make_train_cfg(cfg,"joystick",0),None,device=gs.device)
    runner.load(checkpoint,load_optimizer=False); policy=runner.get_inference_policy(device=gs.device)
    env.set_fixed_command(0,yaw=0); obs,_=env.reset()
    print("Joystick:",stick.get_name(),"| up: forward, down: reverse, left/right: turn; Ctrl+C quits",flush=True)
    deadline=time.monotonic()
    try:
        with torch.inference_mode():
            while True:
                pygame.event.pump()
                if not stick.get_attached():
                    env.set_fixed_command(0,yaw=0)
                    print("Joystick disconnected: command set to zero; preview exiting.",flush=True)
                    break
                vx,yaw=command_from_axes(stick.get_axis(a.axis_x),stick.get_axis(a.axis_y),a.deadzone)
                env.set_fixed_command(vx,yaw=yaw)
                env._update_gait(); env._compute_obs(advance_history=False); obs,_=env.get_observations()
                obs,_,_,_=env.step(policy(obs))
                if env.presentation is not None: env.presentation.update()
                deadline+=env.dt
                time.sleep(max(0.0,deadline-time.monotonic()))
                if time.monotonic()-deadline>1: deadline=time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        env.set_fixed_command(0,yaw=0); pygame.quit()
    return 0

if __name__=="__main__":
    raise SystemExit(main())
