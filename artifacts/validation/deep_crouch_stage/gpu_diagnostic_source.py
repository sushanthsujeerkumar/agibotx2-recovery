#!/usr/bin/env python3
"""One bounded deterministic GPU actor diagnostic; no optimization or training."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from x2_recovery.common import CONTROL_DT, MODEL_PATH
from x2_recovery.env import X2RecoveryEnv


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def counts(rows):
    return {
        "completed_episodes": len(rows),
        "clean_stage_successes": sum(row["clean_stage_success"] for row in rows),
        "original_standing_attainments": sum(row["original_standing_attained"] for row in rows),
        "timeouts": sum(row["timeout"] for row in rows),
        "invalid_episodes": sum(row["invalid"] for row in rows),
        "limit_faults": sum(row["limit_fault"] for row in rows),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actor", type=Path, default=Path("artifacts/experiments/crouch_ppo/actor.pt"))
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/validation/crouch_stage/gpu_deterministic"))
    parser.add_argument("--seed", type=int, default=6001)
    parser.add_argument("--start", choices=['crouch','deep_crouch'], default='crouch')
    parser.add_argument("--action-noise-std", type=float, default=0.)
    args = parser.parse_args()
    if args.action_noise_std < 0: parser.error('noise must be nonnegative')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "summary.json").exists():
        parser.error("Choose a fresh output directory")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    actor_digest = digest(args.actor)
    started = time.monotonic()
    env = X2RecoveryEnv(num_envs=16, seed=args.seed, device="cuda:0", reward_version=3,
                        physics_profile="guarded_v2", reset_mode=args.start)
    records = []
    first_results = {}
    episode_indices = [0] * 16
    control_steps = 0
    error = None
    root = Path(__file__).resolve().parents[1]
    try:
        actor = torch.jit.load(str(args.actor), map_location=env.device).eval()
        initial_pool_indices = (env.qpos[:, None, :] - env.reset_q[None, :, :]).abs().amax(-1).argmin(-1).tolist()
        initial_heights = env.qpos[:, 2].tolist()
        config = {
            "task": f"GPU {args.start} diagnostic; NOT ground recovery",
            "controller": "frozen TorchScript actor; optional declared Gaussian action noise; no optimizer or parameter updates",
            "action_noise_std": args.action_noise_std,
            "actor": str(args.actor), "actor_sha256": actor_digest,
            "environment": env.cfg, "num_envs": 16, "maximum_control_steps": 750,
            "maximum_simulated_seconds_per_env": 15., "seed": args.seed,
            "initial_pool_indices": initial_pool_indices,
            "initial_reset_seeds": [args.seed + index for index in initial_pool_indices],
            "initial_pelvis_heights_m": initial_heights,
            "counting": "Read completed-episode log tensors aligned to nonzero(done); separate initial cohort from automatic resets",
            "success_semantics": "GPU training reward-v3 clean-stage termination; touch sensors and geometric bracing proxy, not exact CPU contact audit",
            "source_sha256": {str(path.relative_to(root)): digest(path) for path in
                              (Path(__file__), MODEL_PATH, root / "src/x2_recovery/env.py",
                               root / "src/x2_recovery/common.py", root / "src/x2_recovery/physics.py")},
        }
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        obs = env.get_observations()
        for step in range(750):
            with torch.inference_mode():
                actions = actor(obs["actor"])
            if actions.shape != (16, 31) or not torch.isfinite(actions).all():
                raise RuntimeError("Actor returned invalid shape or nonfinite actions")
            if args.action_noise_std:
                actions = actions + torch.randn_like(actions) * args.action_noise_std
            obs, _rewards, done, extras = env.step(actions)
            control_steps = step + 1
            ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
            if ids:
                logs = extras["log"]
                required = ["recovery/success_rate", "stance/clean_success_rate",
                            "recovery/invalid_rate", "physics/limit_failure_rate",
                            "recovery/original_recovery_rate", "recovery/episode_seconds"]
                for key in required:
                    if key not in logs or logs[key].numel() != len(ids):
                        raise RuntimeError(f"Completed-episode metric missing or misaligned: {key}")
                values = {key: value.detach().cpu().flatten().tolist() for key,value in logs.items()}
                timeouts = extras["time_outs"].detach().cpu().tolist()
                for local_index, env_id in enumerate(ids):
                    clean = bool(values["stance/clean_success_rate"][local_index])
                    invalid = bool(values["recovery/invalid_rate"][local_index])
                    limit = bool(values["physics/limit_failure_rate"][local_index])
                    timeout = bool(timeouts[env_id])
                    if clean and (invalid or limit or timeout):
                        raise RuntimeError("Inconsistent completed-episode success and failure flags")
                    row = {
                        "environment_id": env_id, "episode_index": episode_indices[env_id],
                        "global_control_step": control_steps,
                        "initial_reset_seed": args.seed + initial_pool_indices[env_id] if episode_indices[env_id] == 0 else None,
                        "clean_stage_success": clean,
                        "original_standing_attained": bool(values["recovery/original_recovery_rate"][local_index]),
                        "invalid": invalid, "limit_fault": limit, "timeout": timeout,
                        "termination": "limit_fault" if limit else "invalid" if invalid else "success" if clean else "timeout" if timeout else "unclassified",
                        "episode_seconds": values["recovery/episode_seconds"][local_index],
                        "completed_episode_metrics": {key: value[local_index] for key,value in values.items()},
                    }
                    records.append(row)
                    if env_id not in first_results:
                        first_results[env_id] = row
                    episode_indices[env_id] += 1
                print(json.dumps({"global_control_step": control_steps, "new_completions": len(ids),
                                  "initial_cohort_completed": len(first_results), "all_completed": counts(records)}), flush=True)
            # env.step resets completed worlds internally. Stop as soon as every
            # initial world has one result; any intervening reset episodes remain
            # labelled separately and never replace the first-attempt result.
            if len(first_results) == 16:
                break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        env.close()
    first = [first_results[index] for index in sorted(first_results)]
    summary = {
        "task": f"GPU {args.start} diagnostic; NOT ground recovery",
        "action_noise_std": args.action_noise_std,
        "actor_sha256": actor_digest, "num_envs": 16,
        "control_steps": control_steps, "maximum_sim_time_per_env_s": control_steps * CONTROL_DT,
        "simulation_budget_env_seconds": 16 * control_steps * CONTROL_DT,
        "wall_seconds": time.monotonic() - started,
        "initial_cohort": counts(first), "initial_cohort_pending": 16 - len(first),
        "all_completed_including_automatic_resets": counts(records),
        "automatic_reset_episode_completions": sum(row["episode_index"] > 0 for row in records),
        "error": error,
        "interpretation": "GPU outcomes use the training environment's own termination metrics. CPU evaluation separately uses exact solver contacts and full-duration stability; backend/criteria differences are not resolved by this one rollout.",
        "initial_cohort_results": first, "all_completed_episode_results": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({key: value for key,value in summary.items() if not key.endswith("results")}), flush=True)
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
