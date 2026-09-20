# Imperfect clock-based teacher augmentation

This dataset adds off-reference observations from a **blended scripted/learned
controller**. It does not establish actor-only success or supine recovery.
The teacher label remains the original time-indexed crouch-to-standing command;
it is not a state-feedback expert and may be an inappropriate corrective action
after a substantial deviation. This is a bounded DAgger-style experiment with
that limitation explicitly retained.

Eight 10 s rollouts used development seeds 121–124, once with teacher weight 0.9
and once with teacher weight 0.6. All eight completed, finished in clean standing,
held clean standing for at least 2 s, and respected trajectory limits. The
learned part was the initially fitted `crouch_teacher/actor.pt`; its identity and
all source hashes are recorded in [config.json](config.json). No training or
independent physical-test seeds were used by this script.

[training_pairs.npz](training_pairs.npz) contains 12,000 samples and 24 episodes:
the original 8,000 scripted samples plus 4,000 blended-rollout samples. The
compatible training interface remains `observations`, `actions`, and per-sample
seeds computed as `seeds[episode_ids]`. **`actions` are the teacher labels;
`executed_actions` are the commands actually applied to the simulator.** The
last 31 observation values correctly contain the previous executed action,
starting from zeros at reset. Episode IDs and offsets are remapped in the
combined dataset; repeated seeds at different teacher weights remain distinct
episodes.

Additional per-sample fields are `time_deltas_s` and `next_state_limits_ok`.
Additional per-episode fields are `teacher_weights`, `source_kind`,
`episode_end_reasons` and `actor_sha256`. The script stops at the first
physics-step limit violation, nonfinite state, pelvis height below 0.3 m, or
episode duration. It retains finite transitions up to that boundary and records
the failure/truncation; these particular eight episodes were not truncated.
Joint positions and velocities are never clipped, and collisions remain enabled.

The limit audit runs at every 1 ms physics step. Clean-standing diagnostics are
sampled every 20 ms using exact ground forces and the existing stance checks.
Previous-action alignment and adjacent-observation equality were asserted for
every combined episode. Full results are in [summary.json](summary.json).

Use a stratified development-validation split. Holding out the last four unique
seeds would reserve all of 121–124, leaving no augmented examples for fitting.
For example, reserving seeds 113–116 and 123–124 retains new seeds 121–122 in
training while keeping both kinds of development data in validation.

Reproduce from the repository root, choosing a fresh output directory:

```bash
.venv/bin/python scripts/augment_crouch_teacher.py \
  --actor artifacts/experiments/crouch_teacher/actor.pt \
  --output artifacts/validation/crouch_stage/augmented_reproduction \
  --teacher-weight .9 .6
```

The dataset SHA-256 is
`700e7b2beecbc9c31d69a3524503f5f3aedf2ef941aa446144b3f5e66ca9eb0e`.
