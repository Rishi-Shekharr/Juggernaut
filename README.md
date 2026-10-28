# Juggernaut

Juggernaut is an eight-joint biped robotics project combining 3D-printed mechanical design, MuJoCo/Genesis simulation, reinforcement learning and embedded policy inference on ESP32. A PPO-trained multilayer perceptron controls leg joints using IMU and joint-position observations.

## Repository contents

- `JUGGERNAUT-v7/`: CAD-derived MuJoCo model, meshes, simulation environment, staged PPO training and evaluation code. The `MODEL.md` and `TRAINING.md` files explain the model and training configuration.
- `JUGGERNAUT-v7-joystick-finetune/`: a warm-started curriculum for forward, reverse, turning and stop commands, plus an interactive Genesis joystick preview.
- `esp32-inference-benchmark/`: ESP32 benchmark sketch with a quantized first layer and policy weights.
- `JUGGERNAUT-v7/logs/juggernaut_v7_realistic/20261027_140516_559717/`: selected walking checkpoint and the configuration, model and source snapshot needed to inspect or warm-start it.

## Setup

Use Python 3.11 and a compatible CUDA-enabled PyTorch installation for GPU training. Install the dependencies in `JUGGERNAUT-v7/requirements-training.txt`; install `pygame` for joystick input. Genesis and PyTorch support depends on the target platform and GPU.

```bash
python -m venv .venv
# Activate the virtual environment for your platform.
python -m pip install -r JUGGERNAUT-v7-joystick-finetune/requirements-training.txt
```

To start joystick curriculum training from the included checkpoint:

```bash
cd JUGGERNAUT-v7-joystick-finetune
python pipeline.py --backend auto
```

The default pipeline uses 2,048 parallel environments and 8,000 PPO updates. Use `--num_envs` to reduce memory requirements. It writes new logs and evaluates the resulting checkpoint; runtime depends on the hardware. After training, use `python drive.py --run RUN_FOLDER --ckpt CHECKPOINT_NUMBER` for the Genesis viewer. For staged training from scratch, see `JUGGERNAUT-v7/TRAINING.md`.

## Embedded benchmark

The ESP32 benchmark uses first-layer INT8 weight quantization with floating-point computation in the other layers. An ESP32-D0WD-V3 measured approximately 10.60 ms mean inference over 1,000 calls. See `esp32-inference-benchmark/README.md` for scope and build instructions.

## Project notes

The simulator uses estimated mass and actuator properties; transfer to a physical robot requires calibration of sensors, servos and timing. Training and checkpoint evaluation produce their own reports; inspect those reports for behavior and acceptance criteria.
