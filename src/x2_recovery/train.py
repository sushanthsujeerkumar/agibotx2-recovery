"""Local PPO training with resumable checkpoints and inexpensive progress files.

Uses the RSL-RL 5.4/5.5 API directly. TensorBoard runs locally; no remote account
or external experiment-logging service is used.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import importlib.metadata
import io
import json
import math
import os
import random
import signal
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner


def ppo_config(seed: int = 0, steps_per_env: int = 24, save_interval: int = 50,
               variant: str = "baseline") -> dict:
    """Conservative feed-forward PPO defaults, shared by training and inference."""
    if variant not in {"baseline", "stability", "stance", "balance"}:
        raise ValueError(f"Unknown PPO variant: {variant}")
    model = {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128, 128],
        "activation": "elu",
        "obs_normalization": True,
    }
    cfg = {
        "variant": variant,
        "seed": seed,
        "num_steps_per_env": steps_per_env,
        "save_interval": save_interval,
        "obs_groups": {"actor": ["actor"], "critic": ["critic"]},
        "logger": "tensorboard",
        "check_for_nan": True,
        "actor": {
            **copy.deepcopy(model),
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 0.6,
                "std_type": "log",
            },
        },
        "critic": copy.deepcopy(model),
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "entropy_coef": 0.01,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "normalize_advantage_per_mini_batch": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }
    if variant == "stability":
        cfg["actor"]["distribution_cfg"].update(init_std=0.4, std_range=(0.05, 0.6))
        cfg["algorithm"]["entropy_coef"] = 1e-4
    elif variant in {"stance", "balance"}:
        cfg["actor"]["distribution_cfg"].update(init_std=0.10, std_range=(0.03, 0.20))
        cfg["algorithm"].update(entropy_coef=1e-4, learning_rate=5e-5,
                                schedule="fixed", clip_param=0.1)
        if variant == "balance":
            cfg["actor"]["distribution_cfg"].update(init_std=.03, std_range=(.01, .08))
            cfg["algorithm"]["entropy_coef"] = 0.
    return cfg


def _json_default(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.ndarray, np.generic)):
        return value.tolist()
    if isinstance(value, (Path, torch.device)):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=_json_default, allow_nan=False) + "\n")
    temporary.replace(path)


def _mean(value: Any) -> float | None:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if not tensor.numel():
        return None
    result = float(tensor.mean().item())
    if not math.isfinite(result):
        raise FloatingPointError("A logged training metric is nonfinite; inspect the environment and PPO update")
    return result


def export_actor(checkpoint_path: str | Path, output_path: str | Path) -> Path:
    """Export a trusted checkpoint as a CPU-only TorchScript policy.

    The artifact accepts concatenated actor observations [batch, num_obs] and
    returns deterministic actions. Its observation normalizer is embedded;
    deployment needs PyTorch, but does not need RSL-RL or TensorDict imports.
    """
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["train_cfg"]
    actor_config = copy.deepcopy(config["actor"])
    if actor_config.pop("class_name") != "MLPModel":
        raise ValueError("Only the project's feed-forward MLP policy is supported for export")
    observations = TensorDict({key: torch.zeros(1, *shape)
                              for key, shape in checkpoint["observation_shapes"].items()}, batch_size=[1])
    actor = MLPModel(observations, config["obs_groups"], "actor", checkpoint["num_actions"], **actor_config)
    actor.load_state_dict(checkpoint["actor_state_dict"])
    scripted = torch.jit.script(actor.as_jit().cpu().eval())
    destination = Path(output_path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    scripted.save(str(temporary))
    temporary.replace(destination)
    return destination


class TrainingStopped(Exception):
    """Stop requested after a complete PPO update, preserving a valid checkpoint."""


class RecoveryRunner(OnPolicyRunner):
    """Upstream PPO loop plus atomic checkpoints and one JSON record per update."""

    def __init__(self, env: Any, train_cfg: dict, log_dir: str, device: str,
                 max_seconds: float = 0.0, print_interval: int = 10) -> None:
        super().__init__(env, copy.deepcopy(train_cfg), log_dir, device)
        # RSL-RL 5.4 destructively pops class names while constructing models.
        # Preserve a complete configuration for checkpoint export and resume.
        resolved_groups = self.cfg["obs_groups"]
        multi_gpu = self.cfg["multi_gpu"]
        self.cfg = copy.deepcopy(train_cfg)
        self.cfg["obs_groups"] = resolved_groups
        self.cfg["multi_gpu"] = multi_gpu
        self.cfg["algorithm"].setdefault("rnd_cfg", None)
        self.cfg["algorithm"].setdefault("symmetry_cfg", None)
        self.logger.cfg = self.cfg
        self.output = Path(log_dir)
        self.output.mkdir(parents=True, exist_ok=True)
        self.completed_iterations = 0
        self.stop_requested = False
        self.max_seconds = max_seconds
        self.print_interval = print_interval
        self.started_at = time.monotonic()
        self.elapsed_before_resume = 0.0
        self.last_progress: dict = {}
        self.last_checkpoint: str | None = None
        self.initialization: dict | None = None
        self._base_log = self.logger.log
        self.logger.log = self._log_progress

    def elapsed_seconds(self) -> float:
        return self.elapsed_before_resume + time.monotonic() - self.started_at

    def status(self, state: str, **extra: Any) -> None:
        _atomic_json(self.output / "status.json", {
            **self.last_progress, "state": state, "pid": os.getpid(),
            "updated_unix": time.time(), "wall_time_s": self.elapsed_seconds(),
            "checkpoint": self.last_checkpoint, **extra,
        })

    def _log_progress(self, **kwargs: Any) -> None:
        # Snapshot extras before the stock logger clears its episode buffer.
        metrics: dict[str, float | None] = {}
        for key in dict.fromkeys(key for item in self.logger.ep_extras for key in item):
            values = [torch.as_tensor(item[key], device=self.device).flatten()
                      for item in self.logger.ep_extras if key in item]
            metrics[key] = _mean(torch.cat(values))
        iteration = int(kwargs["it"]) + 1
        if iteration == 1 or iteration % self.print_interval == 0:
            self._base_log(**kwargs)
        else:
            with contextlib.redirect_stdout(io.StringIO()):
                self._base_log(**kwargs)
        self.completed_iterations = iteration
        duration = kwargs["collect_time"] + kwargs["learn_time"]
        self.last_progress = {
            "iteration": iteration,
            "environment_steps": int(self.logger.tot_timesteps),
            "wall_time_s": self.elapsed_seconds(),
            "training_time_s": float(self.logger.tot_time),
            "iteration_time_s": duration,
            "steps_per_second": self.env.num_envs * self.cfg["num_steps_per_env"] / max(duration, 1e-9),
            "mean_episode_reward": statistics.mean(self.logger.rewbuffer) if self.logger.rewbuffer else None,
            "mean_episode_length": statistics.mean(self.logger.lenbuffer) if self.logger.lenbuffer else None,
            "success_rate": metrics.get("recovery/success_rate"),
            "learning_rate": float(kwargs["learning_rate"]),
            "mean_action_std": _mean(kwargs["action_std"]),
            "losses": {key: _mean(value) for key, value in kwargs["loss_dict"].items()},
            "metrics": metrics,
        }
        with (self.output / "progress.jsonl").open("a") as stream:
            stream.write(json.dumps(self.last_progress, allow_nan=False) + "\n")
        self.status("RUNNING")
        if self.stop_requested or (self.max_seconds > 0 and time.monotonic() - self.started_at >= self.max_seconds):
            raise TrainingStopped

    def save(self, path: str, infos: dict | None = None) -> None:
        # Norm statistics are buffers within actor_state_dict/critic_state_dict.
        checkpoint = self.alg.save()
        checkpoint.update({
            "format_version": 1,
            "iter": self.completed_iterations,
            "infos": infos or {},
            "train_cfg": self.cfg,
            "environment_cfg": self.env.cfg,
            "initialization": self.initialization,
            "observation_shapes": {key: list(value.shape[1:]) for key, value in self.env.get_observations().items()},
            "num_actions": self.env.num_actions,
            "completed_iterations": self.completed_iterations,
            "environment_steps": self.logger.tot_timesteps,
            "training_time_s": self.logger.tot_time,
            "wall_time_s": self.elapsed_seconds(),
            "rng_state": {
                "python": random.getstate(), "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        })
        if callable(getattr(self.env, "state_dict", None)):
            checkpoint["environment_state"] = self.env.state_dict()
        elif hasattr(self.env, "common_step_counter"):
            checkpoint["common_step_counter"] = self.env.common_step_counter
        destination = Path(path)
        # Upstream names files with zero-based iterations; use completed counts.
        if destination.name.startswith("model_"):
            destination = destination.with_name(f"model_{self.completed_iterations:06d}.pt")
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(destination)
        latest = self.output / "latest.pt"
        latest_temporary = self.output / ".latest.pt.tmp"
        latest_temporary.unlink(missing_ok=True)
        latest_temporary.symlink_to(os.path.relpath(destination, self.output))
        latest_temporary.replace(latest)
        # Checkpoints and exports are both replaced atomically for a concurrent viewer.
        exported = destination.with_name(destination.stem + "_actor.pt")
        export_actor(destination, exported)
        actor_latest = self.output / "actor_latest.pt"
        actor_temporary = self.output / ".actor_latest.pt.tmp"
        actor_temporary.unlink(missing_ok=True)
        actor_temporary.symlink_to(os.path.relpath(exported, self.output))
        actor_temporary.replace(actor_latest)
        _atomic_json(self.output / "actor_latest.json", {
            "iteration": self.completed_iterations,
            "environment_steps": self.logger.tot_timesteps,
            "checkpoint": str(destination.resolve()),
            "actor": str(exported.resolve()),
        })
        self.last_checkpoint = str(destination.resolve())
        self.status("RUNNING")

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None) -> dict:
        # Only load trusted project checkpoints: torch pickle is executable.
        checkpoint = torch.load(path, weights_only=False, map_location=map_location or self.device)
        self.alg.load(checkpoint, load_cfg, strict)
        self.initialization = copy.deepcopy(checkpoint.get("initialization"))
        self.completed_iterations = int(checkpoint.get("completed_iterations", checkpoint["iter"]))
        self.current_learning_iteration = self.completed_iterations
        self.logger.tot_timesteps = int(checkpoint.get("environment_steps", 0))
        self.logger.tot_time = float(checkpoint.get("training_time_s", 0))
        self.elapsed_before_resume = float(checkpoint.get("wall_time_s", 0))
        if "environment_state" in checkpoint and callable(getattr(self.env, "load_state_dict", None)):
            self.env.load_state_dict(checkpoint["environment_state"])
        elif "common_step_counter" in checkpoint and hasattr(self.env, "common_step_counter"):
            self.env.common_step_counter = checkpoint["common_step_counter"]
        rng = checkpoint.get("rng_state", {})
        if "python" in rng:
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if rng["cuda"] and torch.cuda.is_available():
                for device_index, state in enumerate(rng["cuda"][:torch.cuda.device_count()]):
                    torch.cuda.set_rng_state(state.cpu(), device=device_index)
        self.last_checkpoint = str(Path(path).resolve())
        return checkpoint.get("infos", {})

    def initialize_from(self, path: str | Path) -> dict:
        """Initialize a fresh experiment from trusted learned networks only.

        Copy actor mean/normalization and critic/normalization while keeping the
        new run's Gaussian distribution parameters, optimizer, RNG and counters.
        Architecture and observation ordering must match; reward version may differ.
        """
        if (self.completed_iterations or self.current_learning_iteration
                or self.logger.tot_timesteps or self.alg.optimizer.state
                or self.initialization is not None):
            raise ValueError("initialize_from is only valid on a fresh, untrained runner")
        source = Path(path).expanduser().resolve(strict=True)
        # Hash and load the same open file, even if another process replaces a symlink.
        with source.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            checkpoint = torch.load(stream, weights_only=False, map_location=self.device)
        parent_cfg = checkpoint["train_cfg"]
        for model_name in ("actor", "critic"):
            parent_model = {k: v for k, v in parent_cfg[model_name].items() if k != "distribution_cfg"}
            current_model = {k: v for k, v in self.cfg[model_name].items() if k != "distribution_cfg"}
            if parent_model != current_model:
                raise ValueError(f"Parent {model_name} architecture/normalization differs from the new run")
        if parent_cfg["obs_groups"] != self.cfg["obs_groups"]:
            raise ValueError("Parent observation-group order differs from the new run")
        shapes = {key: list(value.shape[1:]) for key, value in self.env.get_observations().items()}
        if checkpoint["observation_shapes"] != shapes or checkpoint["num_actions"] != self.env.num_actions:
            raise ValueError("Parent observation/action dimensions differ from the new environment")
        actor_state = self.alg.get_policy().state_dict()
        mean_keys = {key for key in actor_state if not key.startswith("distribution.")}
        parent_mean = {key: value for key, value in checkpoint["actor_state_dict"].items()
                       if not key.startswith("distribution.")}
        if set(parent_mean) != mean_keys:
            raise ValueError("Parent actor mean/normalizer state is incompatible")
        actor_state.update(parent_mean)
        self.alg.load({"actor_state_dict": actor_state,
                       "critic_state_dict": checkpoint["critic_state_dict"]},
                      {"actor": True, "critic": True, "optimizer": False,
                       "iteration": False, "rnd": False}, strict=True)
        self.initialization = {
            "parent_checkpoint": str(source), "parent_sha256": digest,
            "parent_completed_iterations": int(checkpoint.get("completed_iterations", checkpoint["iter"])),
            "parent_environment_steps": int(checkpoint.get("environment_steps", 0)),
            "parent_variant": parent_cfg.get("variant", "baseline"),
            "parent_reward_version": checkpoint.get("environment_cfg", {}).get("reward_version", 1),
            "loaded": ["actor_mean_network", "actor_observation_normalizer",
                       "critic_network", "critic_observation_normalizer"],
            "reset": ["action_distribution", "optimizer", "iteration_and_step_counters",
                      "environment_state", "random_number_generators"],
            "new_run_seed": self.cfg["seed"],
        }
        return copy.deepcopy(self.initialization)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variant", choices=["baseline", "stability", "stance", "balance"], default="baseline",
                        help="Configuration for a new run; resume uses the saved variant.")
    parser.add_argument("--max-iterations", type=int, default=3000, help="Total target, including iterations already trained.")
    parser.add_argument("--steps-per-env", type=int, default=24)
    parser.add_argument("--save-interval", type=int, default=50)
    parser.add_argument("--print-interval", type=int, default=10)
    parser.add_argument("--max-seconds", type=float, default=0, help="This invocation's wall-clock budget; 0 means unlimited.")
    parser.add_argument("--output", type=Path, default=Path("runs/local"))
    continuation = parser.add_mutually_exclusive_group()
    continuation.add_argument("--resume", type=Path)
    continuation.add_argument("--initialize-from", type=Path,
                              help="Start a new experiment using only parent actor/critic networks and normalizers.")
    parser.add_argument("--env-kwargs", default="{}", help="Additional environment constructor arguments as JSON.")
    parser.add_argument("--torch-threads", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = make_parser().parse_args(argv)
    for name in ("num_envs", "max_iterations", "steps_per_env", "save_interval", "print_interval", "torch_threads"):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.max_seconds < 0:
        raise SystemExit("--max-seconds must be nonnegative")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Fix the CUDA/PyTorch installation, or explicitly select --device cpu.")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "progress.jsonl").exists() and args.resume is None:
        raise SystemExit(f"{output} already contains a run; use --resume or choose a new --output directory")
    if args.initialize_from and args.initialize_from.expanduser().resolve().parent == output:
        raise SystemExit("Initialization output must differ from the parent checkpoint directory")
    env_kwargs = json.loads(args.env_kwargs)
    if not isinstance(env_kwargs, dict) or any(key in env_kwargs for key in ("num_envs", "device", "seed")):
        raise SystemExit("--env-kwargs must be an object without num_envs, device or seed")
    torch.set_num_threads(args.torch_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = ppo_config(args.seed, args.steps_per_env, args.save_interval, args.variant)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        cfg = copy.deepcopy(checkpoint["train_cfg"])
        cfg["save_interval"] = args.save_interval
        cfg["logger"] = "tensorboard"
        saved_reward_version = checkpoint.get("environment_cfg", {}).get("reward_version", 1)
        if env_kwargs.get("reward_version", saved_reward_version) != saved_reward_version:
            raise SystemExit("Resume must preserve checkpoint reward_version; start a separate run for a new experiment")
        env_kwargs["reward_version"] = saved_reward_version
        for key, default in [("physics_profile", "legacy"), ("reset_mode", "supine")]:
            saved = checkpoint.get("environment_cfg", {}).get(key, default)
            if env_kwargs.get(key, saved) != saved:
                raise SystemExit(f"Resume must preserve {key}; initialize a separate experiment instead")
            env_kwargs[key] = saved
        del checkpoint
    elif args.variant in {"stability", "stance", "balance"}:
        reward_version = {"stability": 2, "stance": 3, "balance": 3}[args.variant]
        if env_kwargs.get("reward_version", reward_version) != reward_version:
            raise SystemExit(f"The {args.variant} variant requires reward_version={reward_version}")
        env_kwargs["reward_version"] = reward_version
        if args.variant == "balance":
            for key, required in [("physics_profile", "guarded_v2"), ("reset_mode", "balance")]:
                if env_kwargs.get(key, required) != required:
                    raise SystemExit(f"Balance variant requires {key}={required}")
                env_kwargs[key] = required
    from x2_recovery.env import X2RecoveryEnv

    _atomic_json(output / "status.json", {
        "state": "STARTING", "pid": os.getpid(), "updated_unix": time.time(),
        "num_envs": args.num_envs, "device": args.device,
    })
    env: Any = None
    runner: RecoveryRunner | None = None
    try:
        env = X2RecoveryEnv(num_envs=args.num_envs, device=args.device, seed=args.seed, **env_kwargs)
        runner = RecoveryRunner(env, cfg, str(output), args.device, args.max_seconds, args.print_interval)
        runner.add_git_repo_to_log(__file__)
        if args.resume:
            runner.load(str(args.resume))
        elif args.initialize_from:
            runner.initialize_from(args.initialize_from)
        elif args.variant == "balance":
            # Begin from the independently validated nominal pose controller.
            # PPO still samples actions and learns; do not credit this prior as learning.
            layers = [m for m in runner.alg.get_policy().modules() if isinstance(m, torch.nn.Linear)]
            if not layers or layers[-1].out_features != env.num_actions:
                raise RuntimeError("Cannot identify actor mean output layer")
            torch.nn.init.zeros_(layers[-1].weight)
            torch.nn.init.zeros_(layers[-1].bias)
            runner.initialization = {"method": "zero_mean_nominal_pose_prior",
                                     "purpose": "Preserve validated standing controller at initialization",
                                     "trained_recovery": False}
        versions = {}
        for package in ("torch", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib", "mjlab"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        _atomic_json(output / "config.json", {
            "training": cfg, "environment": env.cfg, "arguments": vars(args),
            "initialization": runner.initialization,
            "versions": versions, "started_unix": time.time(),
        })
        def request_stop(signum: int, _frame: Any) -> None:
            print(f"Signal {signum}: will save and stop after the current PPO update.", flush=True)
            runner.stop_requested = True
        for signum in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signum, request_stop)
        remaining = args.max_iterations - runner.completed_iterations
        runner.status("RUNNING")
        if remaining > 0:
            runner.learn(remaining, init_at_random_ep_len=False)
        runner.save(str(output / f"model_{runner.completed_iterations:06d}.pt"))
        runner.status("COMPLETED")
    except TrainingStopped:
        assert runner is not None
        runner.save(str(output / f"model_{runner.completed_iterations:06d}.pt"))
        runner.status("PAUSED")
    except BaseException as error:
        if runner is not None:
            runner.status("FAILED", error=f"{type(error).__name__}: {error}")
        else:
            _atomic_json(output / "status.json", {
                "state": "FAILED", "pid": os.getpid(), "updated_unix": time.time(),
                "error": f"{type(error).__name__}: {error}",
            })
        raise
    finally:
        if runner is not None and runner.logger.writer is not None:
            runner.logger.writer.close()
        if callable(getattr(env, "close", None)):
            env.close()


if __name__ == "__main__":
    main()
