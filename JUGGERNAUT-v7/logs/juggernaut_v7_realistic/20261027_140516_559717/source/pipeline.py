"""Train/gate portable V7 v4 stages. Expanded pitch commands require a fresh policy."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from stages import STAGES, DEFAULT_BUDGETS, PIPELINE_VERSION, REWARD_VERSION
from portability import path_reference, resolve_reference, read_stage_paths

ROOT = Path(__file__).resolve().parent

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    temp.replace(path)

def fingerprint():
    files = [path for path in ROOT.iterdir() if path.is_file() and path.suffix in (".py", ".yaml", ".xml", ".json")]
    files += [path for path in (ROOT/"meshes").rglob("*") if path.is_file()]
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(files)}

def positive(value):
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return n

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resume", type=Path, help="v4 pipeline directory or pipeline.json, including after copying the whole folder")
    p.add_argument("--init_actor_checkpoint", type=Path, help="optional compatible V4 actor weights for a NEW bootstrap; critic/optimizer/iteration reset")
    p.add_argument("--num_envs", type=positive)
    p.add_argument("--backend", choices=("auto", "torch", "warp"))
    p.add_argument("--cpu", action="store_true", default=None)
    p.add_argument("--seed", type=int)
    p.add_argument("--log_root", type=Path)
    p.add_argument("--iterations", action="append", default=[], metavar="STAGE=COUNT")
    p.add_argument("--gate_envs", type=positive)
    a = p.parse_args()
    overrides = {}
    for item in a.iterations:
        try:
            stage, number = item.split("=")
            if stage not in STAGES or int(number) < 1:
                raise ValueError()
            overrides[stage] = int(number)
        except ValueError:
            p.error("--iterations needs a V7 STAGE=positive_count")
    defaults = dict(num_envs=2048, backend="auto", cpu=False, seed=1, gate_envs=32)
    if a.resume:
        path = a.resume.resolve()
        path = path/"pipeline.json" if path.is_dir() else path
        state = read(path)
        if (state.get("pipeline_version") != PIPELINE_VERSION
                or state.get("reward_version") != REWARD_VERSION or state.get("format_version") != 2):
            p.error("This is an incompatible pipeline action/reward contract. Start a fresh v4 pipeline; old v1/v2/v3 checkpoints are incompatible.")
        if state["inputs_sha256"] != fingerprint():
            p.error("V7 source/model/config changed; restore the saved inputs or start a fresh pipeline")
        settings = state["settings"]
        for key in defaults:
            value = getattr(a, key)
            if value is not None and value != settings[key]:
                p.error("--"+key+" differs from the saved pipeline settings")
        if any(value != settings["budgets"][stage] for stage, value in overrides.items()):
            p.error("Resume reuses exact stage budgets")
        if a.log_root and a.log_root.resolve() != resolve_reference(settings["log_root"], path.parent):
            p.error("--log_root differs from this relocated pipeline's log root")
        if a.init_actor_checkpoint:
            reference = settings.get("init_actor_checkpoint")
            if not reference or a.init_actor_checkpoint.resolve() != resolve_reference(reference, path.parent):
                p.error("Resume reuses the original actor initialization; omit --init_actor_checkpoint")
    else:
        settings = {key: getattr(a, key) if getattr(a, key) is not None else value for key, value in defaults.items()}
        settings["budgets"] = dict(DEFAULT_BUDGETS, **overrides)
        if not 0 <= settings["seed"] < 2**32 or settings["cpu"] and settings["backend"] == "warp":
            p.error("Invalid seed or CPU/Warp combination")
        log_root = (a.log_root or ROOT/"logs").resolve()
        actor_init = a.init_actor_checkpoint.resolve() if a.init_actor_checkpoint else None
        if actor_init and not actor_init.is_file():
            p.error("Actor checkpoint does not exist: "+str(actor_init))
        folder = log_root/"pipelines"/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        folder.mkdir(parents=True, exist_ok=False)
        path = folder/"pipeline.json"
        settings["log_root"] = path_reference(log_root, folder)
        settings["init_actor_checkpoint"] = path_reference(actor_init, folder) if actor_init else None
        settings["init_actor_sha256"] = hashlib.sha256(actor_init.read_bytes()).hexdigest() if actor_init else None
        state = dict(pipeline_version=PIPELINE_VERSION, reward_version=REWARD_VERSION, format_version=2,
                     path_base="manifest_directory", state="created", settings=settings,
                     inputs_sha256=fingerprint(), stages={}, created_local=datetime.now().isoformat())
        write(path, state)
    anchor = path.parent
    log_root = resolve_reference(settings["log_root"], anchor)
    actor_init = resolve_reference(settings["init_actor_checkpoint"], anchor) if settings.get("init_actor_checkpoint") else None
    if actor_init and (not actor_init.is_file() or hashlib.sha256(actor_init.read_bytes()).hexdigest() != settings["init_actor_sha256"]):
        p.error("Actor initialization checkpoint is missing or changed after transfer")
    print("Pipeline:", path, flush=True)
    print("Objective:", REWARD_VERSION, "; initialization:", "actor only" if actor_init else "fresh", flush=True)
    baseline = None
    predecessor = None
    for stage in STAGES:
        previous = state["stages"].get(stage, {})
        if previous.get("state") == "accepted":
            selected = resolve_reference(previous["selected_checkpoint"], anchor)
            if not selected.is_file():
                p.error("Accepted checkpoint missing: "+str(selected))
            predecessor = str(selected)
            if stage == "bootstrap":
                baseline = previous["nominal_speed_m_s"]
            continue
        result_path = anchor/(stage+".json")
        result = read(result_path) if result_path.is_file() else None
        existing, final_checkpoint = read_stage_paths(result, result_path) if result else (None, None)
        budget = settings["budgets"][stage]
        if not result or result.get("state") != "complete" or result.get("completed_iterations", 0) < budget:
            cmd = [sys.executable, str(ROOT/"train.py"), "--stage", stage,
                   "--num_envs", str(settings["num_envs"]), "--max_iterations", str(budget),
                   "--backend", settings["backend"], "--seed", str(settings["seed"]),
                   "--log_root", str(log_root), "--result_json", str(result_path)]
            if settings["cpu"]:
                cmd.append("--cpu")
            if existing and existing.is_dir() and any(existing.glob("model_*.pt")):
                cmd += ["--resume", "--run", existing.name]
            elif predecessor:
                cmd += ["--load_checkpoint", predecessor]
            elif stage == "bootstrap" and actor_init:
                cmd += ["--init_actor_checkpoint", str(actor_init)]
            state.update(state="training", active_stage=stage)
            write(path, state)
            rc = subprocess.run(cmd, cwd=ROOT, check=False).returncode
            if rc:
                state.update(state="interrupted" if rc in (130,-2) else "failed", returncode=rc)
                write(path, state)
                return rc
            result = read(result_path)
            if result.get("state") != "complete" or result.get("completed_iterations", 0) < budget:
                p.error(stage+" did not finish its budget")
        run, final_checkpoint = read_stage_paths(result, result_path)
        if not final_checkpoint or not final_checkpoint.is_file():
            p.error("Completed stage checkpoint missing after transfer")
        candidates = read(run/"candidates.json") if (run/"candidates.json").is_file() else []
        checkpoints = list(dict.fromkeys([str(resolve_reference(item["checkpoint"],run)) for item in candidates]
                                        +[str(final_checkpoint)]))
        if any(not Path(checkpoint).is_file() for checkpoint in checkpoints):
            p.error("A shortlisted checkpoint is missing")
        reports = {}
        for profile in ("nominal","stage"):
            output = anchor/(stage+"_"+profile+"_gates.json")
            cmd = [sys.executable,str(ROOT/"assess.py"),"--run",str(run),"--checkpoints",*checkpoints,
                   "--profile",profile,"--output",str(output),"--num_envs",str(settings["gate_envs"]),
                   "--backend",settings["backend"]]
            if settings["cpu"]:
                cmd.append("--cpu")
            if profile=="nominal" and baseline is not None:
                cmd += ["--baseline_speed",str(baseline)]
            state.update(state="assessing",active_stage=stage)
            write(path,state)
            rc = subprocess.run(cmd,cwd=ROOT,check=False).returncode
            if rc:
                state.update(state="assessment_error",returncode=rc)
                write(path,state)
                return rc
            reports[profile]={row["checkpoint"]:row for row in read(output)["results"]}
        def candidate_quality(checkpoint):
            rows = [reports[profile][checkpoint] for profile in reports]
            trials = [trial for row in rows for trial in row["trials"]]
            # Prefer an actually walking policy when no candidate passes every gate.
            walking_checks = ("completion", "speed", "left_clearance", "right_clearance")
            walking = sum(bool(t["checks"][name]) for t in trials for name in walking_checks)
            checks = sum(sum(bool(value) for value in t["checks"].values()) for t in trials)
            checks += sum(bool(row["nominal_retention_passed"]) for row in rows)
            symmetry = min((t["valid_pair_completion_ratio"] or 0.0)
                           * (t["paired_swing_balance_ratio"] or 0.0)
                           * max(0.0, 1.0-(t["paired_swing_time_gap_fraction"]
                                             if t["paired_swing_time_gap_fraction"] is not None else 1.0))
                           for t in trials)
            speed = min(row["mean_speed_m_s"] for row in rows)
            return (all(row["passed"] for row in rows), walking, checks, symmetry, speed)
        best=max(checkpoints,key=candidate_quality)
        quality=candidate_quality(best)
        nominal_speed=reports["nominal"][best]["mean_speed_m_s"]
        state["stages"][stage]=dict(state="accepted",selected_checkpoint=path_reference(best,anchor),
                                    gates_passed=quality[0],
                                    reports={profile:stage+"_"+profile+"_gates.json" for profile in reports},
                                    nominal_speed_m_s=nominal_speed,run_dir=path_reference(run,anchor),
                                    minimum_symmetry_quality=quality[3])
        if not quality[0]:
            print("Stage has failed checks; continuing with best available checkpoint. Reports:", anchor, flush=True)
        write(path,state)
        predecessor=best
        if stage=="bootstrap":
            baseline=nominal_speed
    state.update(state="complete",active_stage=None,selected_checkpoint=path_reference(predecessor,anchor))
    write(path,state)
    print("All stages completed; inspect gate reports for failed checks. Final checkpoint:",predecessor,flush=True)
    return 0

if __name__=="__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; resume the saved v4 pipeline.json.",flush=True)
        raise SystemExit(130)
