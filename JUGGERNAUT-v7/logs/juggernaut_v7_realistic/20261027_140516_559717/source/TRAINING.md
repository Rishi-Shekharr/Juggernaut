# JUGGERNAUT V7 training pipeline

V7 uses its own CAD export, geometry, estimated component inertials and collision proxies. Start a fresh V7 policy: V5/V6 checkpoints have incompatible geometry or observation contracts. The raw export is preserved in JUGGERNAUT-v7.raw.xml.

The current pipeline builds on the V5 walking rewards with a small torque penalty and paired airborne-time penalty under `v7_timing_effort_ramp_v2`, including V5 swing-pair accounting, the 0.6 s gait period and height-adjusted 0.1655 m base-height target. V7 retains its own geometry, collision proxies and expanded +/-0.9 rad pitch commands. Start a fresh run because the reward contract and actuator model changed. Historical logs, diagnostics and source snapshots are preserved.

The imported v2 run completed 1500 bootstrap updates but failed clearance and gait gates. The later screenshot showed an expanded-pitch run with low lifts and approximately 10-12% valid bilateral pairs, still using the old collision model. These are historical results, not validation of the current reward or actuator settings. See TRANSFER.md for the fresh-run command.

## Run the complete pipeline later

From PowerShell, using the environment with the pinned dependencies:

~~~powershell
Set-Location "YOUR_DESTINATION\JUGGERNAUT-v7"
python .\pipeline.py --num_envs 2048 --backend auto
~~~

Activate the target laptop's Genesis/CUDA Python environment first. The portable launch_training.cmd uses its active Python and accepts the same options. requirements-training.txt specifies Genesis 1.4.0, rsl-rl-lib 2.2.4 and the CUDA/PyTorch stack. Auto selects Warp on CUDA and Torch on CPU. Reduce num_envs if memory is insufficient.

The pipeline creates logs/pipelines/<timestamp>/pipeline.json, freezes each run's source/model/meshes, and runs these stages:

| Stage | PPO updates | Arms | Newly introduced variation |
|---|---:|---|---|
| bootstrap | 2000 | Fixed raised pose | Nominal motors; command expands from 0.10 to 0.15 m/s |
| arm_poses | 650 | One raised pose per episode | Pose range expands after 50 updates over a 300-update ramp |
| arm_motion | 650 | Slow pose changes | Holds for 5 to 8 seconds plus travel and settling |
| response | 1400 | Same | V5 bootstrap dynamics; domain randomization disabled |
| delay | 1200 | Same | V5 bootstrap dynamics; domain randomization disabled |
| realistic | 2600 | Same | V5 realistic-stage domain randomization |

Total: 8500 updates. Each stage defaults to 2048 environments, 24 steps per update and 30-second episodes. Override budgets with an explicit argument such as --iterations bootstrap=2500. A shorter budget may stop before its curriculum finishes and fail the stage gate.

Domain randomization is disabled through bootstrap, arm_poses, arm_motion, response and delay. Only realistic adds V5's realistic-stage domain randomization. Randomized parameters remain fixed within an episode.

All stages retain their walking rewards. During bootstrap, the torque and paired airborne-time penalties are zero for the first 1,000 updates, then increase linearly to full weights over the next 500 updates. They stay at full strength for the final 500 bootstrap updates and all later stages. The ramp fraction is logged as `reward_penalty_alpha`.

~~~powershell
python .\pipeline.py --resume .\logs\pipelines\<timestamp>\pipeline.json
~~~

Resume binds exact settings, source/model fingerprints and run identities. Manifests, results and checkpoint candidates store relative path references and remain usable after the entire folder is copied. Changed inputs require a fresh pipeline or restoring originals. Resume only runs created under the same `v7_timing_effort_ramp_v2` objective and model; start a new pipeline after a reward change; a compatible older actor can be used for initialization as described in TRANSFER.md.

## Checkpoint selection and acceptance

The pipeline assesses a shortlist from the mature part of each stage plus the final checkpoint. It prefers candidates that pass both nominal and full-stage gates. If none pass, it ranks basic walking checks first, then the number of passed checks, worst-trial symmetry, and speed. The training shortlist also favors bilateral lift and symmetry. The final checkpoint is not assumed best.

assess.py is run later, as part of a requested pipeline execution. Defaults: 32 robots, seeds 123/456, commands 0.10/0.15 m/s, 30 seconds per trial. Full-stage evaluation disables training replay and evaluates final configured uncertainty.

Each trial requires:

- At least 95% of trajectories finish without a reset.
- Mean speed at least 85% of command; failed trajectories contribute zero thereafter.
- Both completed swing peaks average at least 15 mm, toward the 25 mm training target.
- At least 90% of scheduled bilateral pairs are valid under the new lift criteria.
- Mean per-pair height gap is at most 5 mm and weaker/stronger peak ratio is at least 0.75.
- Mean relative airborne-time gap is at most 0.25. These metrics are accumulated per pair, so opposite asymmetric episodes cannot cancel.
 - No arm-leg contact at or above the 1 N detection threshold.
- Each leg's actual substep RMS torque at or below the provisional 0.392266 Nm budget.

After bootstrap, nominal mean speed must also remain within 10% of the accepted bootstrap baseline. Gate failures remain recorded in the reports and `gates_passed` is false in the pipeline manifest, but the best available candidate advances to the next stage. A process or assessment error still stops the pipeline. Passing every gate is not guaranteed; inspect the reports before deployment.

Training logs include speed, bilateral swing metrics, actual episode duration, termination counts and all 12 motors' actual substep RMS/peak torque, clipping and stall events. Torque and acceleration statistics cover all motors; the normalized substep torque penalty weight is -0.05, and stall events are telemetry rather than a reward or termination. The v1 and imported v2 training runtimes were exercised. The imported v2 bootstrap and gate results are reviewed in diagnostics/imported_run_review.md. The current reward and actuator revision has not been simulated or trained.

## Reward objective

V7 stands 9.25 mm taller than V5: the authored chassis height is 0.171908560948 m instead of 0.162658560948 m. The base-height reward target is therefore 0.1655 m (V5 0.15625 m + 0.00925 m), preserving the same 6.40856 mm offset below standing height. The V7 fall threshold remains 0.0795 m and the spawn margin remains 3 mm.

All stages retain the walking and swing-pair rewards, with the base-height target adjusted for V7 geometry. A -0.2 paired relative airborne-time-gap penalty and a -0.05 normalized torque-effort penalty target the observed failing checks. The shared reward configuration uses a 0.6 s gait period and a height-adjusted 0.1655 m base-height target. V7 adds no V4 dense-lift term or alternate pair accounting. The V5 configuration and preserved revision notes are the reference for exact reward values; see `diagnostics/lift_reward_changes.md` for historical context.

The pipeline reward version is `v7_timing_effort_ramp_v2`. The existing walking reward definitions and pair bookkeeping are retained; stage-specific reward configuration remains in the corresponding V7 `config*.yaml` files. Episodes terminate for V5 fall conditions. V7 stall and arm-contact detections are telemetry and assessment inputs only; they do not add termination or reward penalties.

## Direct arm control and clearance

The walking actor outputs exactly eight leg actions. Four physical arm servos are controlled directly in this order: left shoulder, left mandible, right shoulder, right mandible.

Call env.set_external_arm_commands with four absolute model-joint radians. This suspends random sampling until reset or env.resume_external_arm_sampling(). The API is ready for an external/Bluetooth command adapter; no Bluetooth transport or physical driver is assumed.

Sampled and manual commands share this walking envelope and a 0.35 rad/s command rate:

| Joint role | Allowed target degrees | Raised reset |
|---|---:|---:|
| Shoulder | -65 to -55 | -60 |
| Mandible | +35 to +55 | +45 |

Requests outside the walking envelope are clipped. Physical joint limits remain broader. No physical arm positions are projected or teleported. All arm-leg collisions remain active. Arm-leg contact is recorded for diagnostics and assessment; it does not independently terminate an episode or change reward. Stage assessment reports detected contact as a failed check.

The raised center has about 12 mm minimum separation between static arm/leg body AABBs. The complete articulated-envelope bound is conservative and does not certify clearance: some swept boxes overlap. See diagnostics/collision_corridor.json. The restricted target envelope, actual-contact checks and future motion gates are all needed. Arbitrary backdrive/falls can still cause contact. Expanding the range for manipulation requires a separate geometry review and later validation.

The assessment uses simulator contact information to identify obstruction. A physical controller still needs an appropriate measured fault response; the actor does not observe contact forces.

## Motor model and deployable observations

Leg actuators: Kp=3.0, Kd=0.06, 0.94 Nm torque cap and 6 rad/s DC no-load speed reference. The 6 rad/s value is a motor-envelope reference, not a physical velocity clamp; loads may back-drive joints faster. Upper shoulder and mandible actuators all use Kp=0.6, Kd=0.01808782111111, 0.1808782111111 Nm torque cap and a 10 rad/s DC no-load speed reference. Their speed is also unrestricted in the physical model.

The realistic stage uses the V5 motor Kp, Kd, torque and speed-envelope scales for all motors; arms do not have separate randomization scales. Speed remains an envelope reference, not an enforced joint-velocity limit. Friction, mass and delay variations are specified in the stage configuration files.

The motor model permits external backdrive. There is no post-physics position/velocity projection or ideal speed constraint. Command slew limits targets only.

The 4 kgf cm continuous estimate is a provisional RMS design budget, separate from peak authority. Gate rollouts measure actual substep RMS; they do not model 20 minutes of heating or certify winding temperature. MG996R PWM accepts position commands; its simulated gain/torque parameters are response-model assumptions, not hardware torque commands.

Actor: five consecutive 46-value frames, oldest first, totaling 230 inputs. Critic: a current 53-value privileged frame. Reset repeats the first new frame five times and clears previous actions.

| Slice per frame | Input |
|---|---|
| 0:3 | Scaled body gyro |
| 3:6 | IMU-fused projected gravity |
| 6:8 | Scaled x/y command |
| 8:20 | Eight measured leg positions, four command-estimated arm positions, relative to defaults and scaled |
| 20:32 | Twelve nominal command-observer velocities |
| 32:40 | Previous clipped leg actions |
| 40:44 | Issued external arm commands relative to defaults, scaled |
| 44:46 | Gait sine/cosine |

The observer uses issued targets, fixed nominal delay and nominal slew only. No randomized delay parameter, true arm position or true shaft velocity enters the actor. The eight leg potentiometers and six-axis IMU must exist on the real robot. Foot contacts, linear velocity and course errors are critic-only; there is no magnetic heading reference.

Control is 20 ms, with ten 2 ms physics steps. Actions clip at +/-1. Pitch commands use action_scale=0.9 rad on all six hip/knee/ankle joints, so the normalized command range requests +/-0.9 rad (about +/-51.6 degrees) of pitch offset. This matches V5/4dof command reach (0.3 rad scale times +/-3 clipping). Roll commands remain at 0.15 rad. The physical pitch target limits are +/-80 degrees (+/-1.3962634 rad): they describe available travel, separate from the commanded range, and match the robot total travel of at least 160 degrees. Hardware signs and PWM zero offsets must be calibrated in model coordinates. The authored left ankle-roll name Revolute-62 is retained.

## Preview and export later

~~~powershell
python .\eval.py --stage realistic --run <run-folder-name> --cmd 0.15 --seconds 30
~~~

Use --no_viewer --record walking.mp4 for recording. --arms_deg -60 45 -60 45 requests an external pose, still guarded. Evaluation prefers the checkpoint's archived runtime/model/meshes.

~~~powershell
python .\export_policy.py --checkpoint <accepted-model.pt> --output .\exports\v7
~~~

Export creates a bounded TorchScript actor, NumPy weights for embedded conversion and deployment.json containing joint order, slices, scales, arm envelope, timing and provenance. It also includes the history helper and software-model sources. Export does not run inference and is not a stage approval.

Before autonomous physical walking, calibrate servo response/latency, assembly mass, IMU axes, potentiometer zeros, power sag and heating. Preserve the guarded arm envelope and history/timing contract in firmware.

## Rebuild the XML

~~~powershell
python .\build_model.py
~~~

This constructs XML/STL collision hulls without launching a simulator. It uses only V7-local raw assets and component_calibration.json; diagnostics/model_manifest.json records assumptions. MODEL.md describes the estimated masses and collision approximations.

The new CAD standing chassis height is 0.171908560948 m; training adds a 3 mm spawn/drop margin. The standalone XML includes a floor, IMU site/sensors and raised stand keyframe. Estimated articulated mass is approximately 1.269 kg. Component masses and effective servo response remain calibration assumptions.
