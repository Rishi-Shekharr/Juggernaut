# Joystick curriculum

This directory fine-tunes the included realistic V7 walking actor for start/stop, forward, reverse and turning commands. The actor produces eight leg targets from joint and IMU observations.

From this directory, install `requirements-training.txt` in a compatible Python 3.11 environment and run `python pipeline.py --backend auto`. The included default warm-start checkpoint is under `../JUGGERNAUT-v7/logs/juggernaut_v7_realistic/20261027_140516_559717/`. Use `--num_envs` for a smaller environment count if needed. The pipeline saves the trained checkpoint and assessment results under `logs/`.

After training, use `python drive.py --run RUN_FOLDER --ckpt CHECKPOINT_NUMBER`. Install pygame for joystick input. The viewer uses the left stick for forward/reverse and yaw commands. See `python drive.py --help` for options.
