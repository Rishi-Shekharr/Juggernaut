# JUGGERNAUT V7 simulation model

The XML is built deterministically from JUGGERNAUT-v7.raw.xml, the local STL assets and component_calibration.json using build_model.py. The build performs static CAD/XML processing without loading a simulator.

The V7 CAD body tree and transforms are retained. The model adds a floating root, floor, chassis IMU sensors and 12 position actuators in joints.yaml order. The stand keyframe uses chassis position (-0.00388449, 0.00000287, 0.17190856) m, quaternion wxyz=(0,0,0,1), zero leg angles, shoulders -60 degrees and mandibles +45 degrees. CAD uppercase XYZ Euler rotations mean fixed axes; SciPy lowercase xyz is equivalent.

## Pitch commands and physical travel

The pipeline keeps eight normalized leg actions clipped to +/-1. Each of the six hip/knee/ankle pitch joints has action_scale=0.9 rad, giving +/-0.9 rad commanded offset (about +/-51.6 degrees). This matches V5/4dof's 0.3 rad scale with +/-3 clipping. Roll remains +/-0.15 rad. Arms have no policy outputs.

The six physical pitch joints and their MJCF position-actuator control ranges use +/-80 degrees: 160 degrees total travel. Roll/shoulder limits remain +/-75 degrees and mandibles +/-80 degrees. Command range and mechanical travel are separate limits. Firmware must use calibrated model zero positions and signs.

V7 uses the V5 reward functions and weights with the V7 height adjustment, including the 0.6 s gait period and height-adjusted 0.1655 m base-height target. Its reward contract is `v5_symmetric_v1`. The V7 geometry and body tree remain V7-specific. This reward revision requires fresh training; historical checkpoints, diagnostics, and source snapshots remain intact.

Static example poses in diagnostics/pitch_workspace.json keep the chassis level at the 0.1655 m reward height and counter-rotate knee/ankle to keep the sole nearly flat. Offsets of +/-0.3 rad yield about 4.0 mm sole clearance, +/-0.6 rad about 17.6 mm, and +/-0.9 rad about 33.0 mm. These examples establish geometric reach, not loaded performance, collision safety or an expected walking result.

## Mass, actuators and collision coverage

Total mass is estimated at 1.268866 kg. MG996R bodies use 55 g, MG90S shoulders 13.4 g, and SG90 mandibles 9 g despite the CAD case labels. The new right hip module uses 76.408 g, including a 55 g servo and 21.408 g bracket. Chassis/battery/electronics/payload is estimated at 370 g. Matching components retain calibrated centers and full inertia tensors. Other known V7 components retain CAD centers and scale all six inertia terms by estimated mass. Unknown components stop construction.

The XML leg actuators use Kp=3, Kd=0.06 and a 0.94 Nm torque cap. Their 6 rad/s DC no-load speed is a reference only: the physical model imposes no speed limit, and external loads can back-drive joints faster. All four upper servos use Kp=0.6, Kd=0.01808782111111, a 0.1808782111111 Nm cap, and a 10 rad/s DC no-load speed reference. Upper speed is also unrestricted by the physical model. The speed values describe the motor envelope; they are not post-physics velocity projections.

Both hips now use the same two-part collision template, reflected between sides. The builder first verifies that the original hip CAD meshes match under that reflection, and records the template source and transform in diagnostics/model_manifest.json. This removes the former oversized right whole-hull proxy and its neutral overlap with the right thigh. Static half-space checks find no positive-volume hip/thigh overlap on either side at zero leg angles. Actual CAD body placements, including the measured small arm/linkage offsets, are retained.

All six paired arm component groups have matching masses. Both printed third links use 2.15762735851 g, replacing the old right-side 2.675953249928 g assignment; all six right-link inertia terms were scaled by the same mass ratio while its center of mass was retained. The builder rejects unequal paired arm masses or a hip hull count other than two per side. Before/after evidence is in diagnostics/model_mirror_fix.json; original diagnostics and XML are preserved in diagnostics/model_fix_before/.

Visual meshes are non-colliding. There are 51 active convex CAD proxies and sole/post boxes. Parent filtering is disabled, and no arm-leg exclusion is introduced. Hulls can bridge openings.

## Arm corridor over the expanded leg range

Arms remain externally commanded inside shoulder [-65,-55] degrees and mandible [35,55] degrees, centered at -60/+45 degrees. Sampled and manual commands share the 0.35 rad/s rate limit. At that center with zero leg angles, the closest arm/leg AABBs remain about 12.3 mm apart.

The rebuilt diagnostics/collision_corridor.json covers pitch +/-1.02 rad and roll +/-0.27 rad: action reach plus 0.02 rad reset noise and 0.10 rad tracking allowance. Angular units are explicit; leg angles above one radian are not interpreted as degrees. Its interval bounds cannot certify separation: 83 of 288 body-pair swept boxes overlap. This means the enclosure cannot prove clearance, not that all those pairs collide. Broad backdrive/fall configurations can still contact.

Physical arm-leg collisions remain active. Contact is recorded for diagnostics and assessment; it does not independently terminate an episode or change reward. Stage assessment can reject detected arm-leg contact. The full motion range needs simulation validation before transfer.

No simulation, training or test suite has been run for this revision. Imported v2 training and gate results are documented in diagnostics/imported_run_review.md.
