"""Path references anchored to the owning manifest/result, portable with the whole folder."""
import os
from pathlib import Path

def path_reference(path, anchor):
    """Prefer relative references; different Windows drives necessarily stay absolute."""
    path, anchor = Path(path).resolve(), Path(anchor).resolve()
    try:
        return Path(os.path.relpath(path, anchor)).as_posix()
    except ValueError:
        return str(path)

def resolve_reference(reference, anchor):
    path = Path(reference)
    return path.resolve() if path.is_absolute() else (Path(anchor)/path).resolve()

def read_stage_paths(result, result_file):
    anchor = Path(result_file).resolve().parent
    if result.get("path_base") != "result_directory":
        raise ValueError("Stage result predates portable V7 v2; start a fresh revised pipeline")
    run = resolve_reference(result["run_dir"], anchor)
    checkpoint = resolve_reference(result["checkpoint"], anchor) if result.get("checkpoint") else None
    return run, checkpoint
