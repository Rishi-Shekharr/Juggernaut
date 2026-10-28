"""Immutable stage order and config filenames shared by the CLI tools."""
STAGES = ("bootstrap", "arm_poses", "arm_motion", "response", "delay", "realistic")
CONFIG_FILES = {stage: ("config.yaml" if stage == "bootstrap" else f"config_{stage}.yaml") for stage in STAGES}
DEFAULT_BUDGETS = dict(zip(STAGES, (2000, 650, 650, 1400, 1200, 2600)))
PIPELINE_VERSION = "juggernaut_v7_external_arms_history_v4"

REWARD_VERSION = "v7_timing_effort_ramp_v2"
