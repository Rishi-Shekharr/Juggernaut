"""Single continuous joystick curriculum; assessment runs after training."""
STAGES = ("maneuver",)
CONFIG_FILES = {"maneuver": "config.yaml"}
DEFAULT_BUDGETS = {"maneuver": 8000}
PIPELINE_VERSION = "juggernaut_v7_joystick_v1"
REWARD_VERSION = "maneuver_body_frame_v1"
