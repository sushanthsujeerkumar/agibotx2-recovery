# Stability-variant verification

The actual GPU smoke run completed five PPO updates in 128 environments (15,360 transitions), using reward version 2 and bounded Gaussian sampling noise. The exported TorchScript actor then ran one complete CPU episode without an invalid state. It timed out with zero successes, as expected for an untrained five-update policy; this is an execution check, not learning evidence.

The separate CPU runner check covers actual PPO updates for both variants, checkpoint configuration, TorchScript inference parity, and effective standard-deviation bounds. The existing three physics-contract tests also passed after the reward-version change.
