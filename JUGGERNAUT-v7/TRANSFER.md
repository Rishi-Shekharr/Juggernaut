# Transfer and start the V7 training pipeline

Copy the entire updated JUGGERNAUT-v7 folder to the training laptop. Keep the current Python sources, six stage configs, joints.yaml, corrected XML, builder, calibration and meshes together. Preserve any historical logs/source snapshots separately or in logs; this update does not rewrite them.

## Start fresh

Activate the Genesis/CUDA Python environment in Command Prompt, then run:

~~~bat
cd /d "YOUR_DESTINATION\JUGGERNAUT-v7"
python pipeline.py --num_envs 2048 --backend auto
~~~

For live console output also saved to training_console.log:

~~~bat
launch_training.cmd --num_envs 2048 --backend auto
~~~

The launcher uses JUGGERNAUT_PYTHON if set, otherwise the active PATH Python or Windows py launcher. Use requirements-training.txt for compatible package versions. If needed, select the interpreter explicitly:

~~~bat
set "JUGGERNAUT_PYTHON=C:\YOUR_ENV\python.exe"
launch_training.cmd --num_envs 2048 --backend auto
~~~

This revision retains the walking rewards and swing-pair accounting, and adds modest airborne-time symmetry and torque-effort penalties (`v7_timing_effort_ramp_v2`), including the 0.6 s gait period and height-adjusted 0.1655 m base-height target. V7 keeps its own geometry, +/-0.9 rad pitch target offsets, +/-80 degree physical pitch travel, matched hip collision hulls, equal paired arm masses and independently commanded arms. Leg actuators use Kp=3, Kd=0.06 and a 0.94 Nm cap; all upper actuators use Kp=0.6, Kd=0.01808782111111 and a 0.1808782111111 Nm cap. The 6 rad/s leg and 10 rad/s upper DC no-load values are references only; the model imposes no velocity limit and permits external back-drive.

## Checkpoint compatibility

Start a new pipeline because the reward contract changed. Optionally initialize only the actor from a compatible saved V7 v5_symmetric_v1 checkpoint: `python pipeline.py --init_actor_checkpoint "logs\juggernaut_v7_bootstrap\20261026_192114_586029\model_1099.pt"`. The critic and optimizer are reset. Bootstrap defaults to 2,000 updates: the first 1,000 use no added penalties, then their weights ramp linearly to full strength across the next 500 and remain there for the final 500. Resume is only for a run created with these same `v7_timing_effort_ramp_v2` files and settings; do not carry over an older objective's value function or optimizer state. Historical checkpoints can still be inspected with their archived runtime and model.

No training or simulation has been run for this revision. The revised objective needs a new training run and later evaluation. Gate failures are reported but do not prevent the pipeline from advancing.

## Resume this run later

After a run has been created with the current files, preserve its logs/pipelines/<timestamp>/pipeline.json and stage logs, then use:

~~~bat
python pipeline.py --resume "logs\pipelines\YOUR_TIMESTAMP\pipeline.json"
~~~

Exact resume retains the saved training settings and source/model fingerprints. Paths are relative, so moving the whole folder preserves them. Do not change environment count, stage budgets, code or configs during an exact resume. External custom asset paths must remain available.
