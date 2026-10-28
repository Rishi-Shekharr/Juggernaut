"""One warm-started training curriculum followed by final soft assessments."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_ACTOR = ROOT.parent / "JUGGERNAUT-v7/logs/juggernaut_v7_realistic/20261027_140516_559717/model_2299.pt"

def save(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--actor", type=Path, default=DEFAULT_ACTOR)
    p.add_argument("--num_envs", type=int, default=2048)
    p.add_argument("--backend", choices=("auto", "torch", "warp"), default="auto")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--gate_envs", type=int, default=16)
    a = p.parse_args()
    if not a.actor.is_file() or a.num_envs < 1 or a.gate_envs < 1 or (a.cpu and a.backend == "warp"):
        p.error("Provide an existing realistic actor, positive environment counts, and a compatible backend")
    start = time.monotonic()
    root = ROOT / "logs"
    folder = root / "pipelines" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    result = folder / "training.json"
    manifest = folder / "pipeline.json"
    state = {"state": "training", "actor_initialization": str(a.actor.resolve()),
             "requested_iterations": 8000, "estimated_seconds_per_iteration": 5}
    save(manifest, state)
    cmd = [sys.executable, str(ROOT/"train.py"), "--stage", "maneuver", "--num_envs", str(a.num_envs),
           "--max_iterations", "8000", "--backend", a.backend, "--log_root", str(root),
           "--init_actor_checkpoint", str(a.actor.resolve()), "--result_json", str(result)]
    if a.cpu: cmd.append("--cpu")
    print("Training 8,000 updates; final assessment runs only after training.", flush=True)
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc:
        state.update(state="training_error", returncode=rc)
        save(manifest,state); return rc
    info = json.loads(result.read_text(encoding="utf-8"))
    if info.get("state") != "complete":
        state.update(state="training_incomplete")
        save(manifest,state); return 1
    run = (result.parent/info["run_dir"]).resolve()
    checkpoint = (result.parent/info["checkpoint"]).resolve()
    state.update(completed_iterations=info["completed_iterations"])
    if not checkpoint.is_file():
        state.update(state="checkpoint_missing")
        save(manifest,state); return 1
    state.update(state="assessing", run_dir=str(run), selected_checkpoint=str(checkpoint))
    save(manifest,state)
    report = folder/"final_gates.json"
    cmd = [sys.executable, str(ROOT/"assess.py"), "--run",str(run),"--checkpoint",str(checkpoint),
           "--output",str(report),"--num_envs",str(a.gate_envs),"--backend",a.backend]
    if a.cpu: cmd.append("--cpu")
    rc = subprocess.run(cmd, cwd=ROOT).returncode
    if rc:
        state.update(state="assessment_error",returncode=rc,elapsed_seconds=round(time.monotonic()-start,1))
        save(manifest,state); return rc
    gates = json.loads(report.read_text(encoding="utf-8"))
    state.update(state="complete",gates_passed=gates["passed"],gate_report=str(report),
                 elapsed_seconds=round(time.monotonic()-start,1))
    save(manifest,state)
    print("Final soft checks passed:",gates["passed"],"Report:",report,flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
