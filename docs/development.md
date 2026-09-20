# Development history

The submission keeps one current model and one set of run instructions. Earlier experimental launchers, checkpoints and long diagnostic outputs have been removed from the working tree to avoid confusing them with the final result. Their committed versions are still in the existing Git history; no history was rewritten.

The main changes during development were:

1. Direct PPO established the simulator and ROS pipeline, but moving and crossed-foot poses exposed weaknesses in the original success criteria.
2. A per-physics-step audit found joint-limit excursions. The guarded controller and explicit stance checks addressed those failure modes.
3. Standing/crouch work and physical trajectory searches supplied a project-generated recovery reference. An external-policy comparison was explored separately and is not used by the final controller.
4. The final experiment fitted a motion prior to eight physical examples, then trained bounded PPO feedback over full supine episodes. The final assessment passed five fresh trials.

For the complete tree before submission cleanup:

```bash
git log --oneline
git show 606f2c0:PHYSICS_V2_RESULTS.md
git show 606f2c0:REFERENCE_STUDENT_RESULTS.md
git show 606f2c0:LANDING_RESIDUAL_RESULTS.md
git show 606f2c0:artifacts/validation/final_reproduction/physics_step_limit_audit.json
```

To inspect all earlier files without changing this checkout:

```bash
git worktree add --detach ../x2-development-history 606f2c0
```

The archive contains the current source and final evidence. The accompanying Git bundle preserves the history if a GitHub clone is unavailable. Uncommitted scratch runs and generated caches were not part of the final reproducibility record and have been removed from the project folder.
