# Preserved stable recovery with crossed stance

Checkpoint9251 passed five fixed-seed episodes (1001–1005) under the original documented success criteria. Both feet support the floor, no other body touches the floor, height/upright/speed conditions hold continuously for two seconds. Time to pass:5.46,4.66,6.00,4.48,6.30 seconds.

This is a real improvement but not a neutral standing posture. The feet reverse their left/right ordering in pelvis coordinates and press against each other. The independent15-second continuation audit verifies signed collision-centre width about -3.2 to -4.5cm at the end, versus +27.4cm in the nominal pose. End hip-yaw angles are approximately -1.24rad left and +1.02rad right. Foot-foot collision is enabled; this is not simply absent collision filtering. Small contact penetration (~0.37mm) reflects compliant solver contacts; visual-mesh intersection still needs checking against the simplified collision geometry.

The reward includes only a weak mean penalty across all31joint deviations. It has no explicit signed stance-width or foot-heading term, and the success criterion does not exclude a crossed stance. These facts explain why this posture is permitted, rather than prove which individual reward term caused it.

Proposed next refinement: retain this checkpoint as a baseline; explicitly reward nominal-range signed stance width and forward foot headings only during the standing phase, discourage foot-foot bracing and extreme hip yaw, and verify collision shapes against visual geometry. Then use a bounded separate fine-tuning experiment and evaluate both the unchanged recovery metric and a separately reported clean-stance metric. Do not silently change the original success definition or forcefully reposition the robot.
