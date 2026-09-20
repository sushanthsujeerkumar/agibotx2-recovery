"""Visible local evaluation; reload newly saved actors between complete episodes."""
import argparse
import json
import time
from pathlib import Path
from .runtime import RecoveryRuntime
from .common import CONTROL_DT, EPISODE_SECONDS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--directory", default="artifacts/runs/local")
    p.add_argument("--minutes", type=float, default=240)
    p.add_argument("--assess-stance", action="store_true")
    args = p.parse_args()
    directory = Path(args.directory)
    actor = None
    runtime = RecoveryRuntime(controller="scripted", render=True, assess_stance=args.assess_stance)
    end = time.monotonic() + args.minutes*60
    episode = 0
    print("VIEWER: scripted baseline until a trained checkpoint is available", flush=True)
    try:
        while time.monotonic() < end and runtime.viewer.is_running():
            manifest = directory / "actor_latest.json"
            if manifest.exists():
                try:
                    meta = json.loads(manifest.read_text())
                    candidate = Path(meta["actor"])
                    stamp = candidate.stat().st_mtime_ns
                    if candidate.exists() and stamp != actor:
                        import torch
                        torch.set_num_threads(2)
                        policy = torch.jit.load(str(candidate), map_location="cpu").eval()
                        runtime.policy = policy
                        runtime.controller = "policy"
                        runtime.display_label = f"PPO checkpoint {meta['iteration']}"
                        actor = stamp
                        print("VIEWER: trained policy", json.dumps(meta), flush=True)
                except (OSError, ValueError, RuntimeError) as exc:
                    print(f"Waiting for readable checkpoint: {exc}", flush=True)
            runtime.reset(seed=1001 + episode % 5)
            for _ in range(round(EPISODE_SECONDS/CONTROL_DT)):
                start = time.monotonic()
                state = runtime.step()
                passed = state['clean_stance_success'] if args.assess_stance else state['success']
                if not runtime.viewer.is_running() or passed or state["invalid"]:
                    break
                time.sleep(max(0., CONTROL_DT-(time.monotonic()-start)))
            print(f"VIEWER episode {episode}: controller={runtime.controller} success={state['success']} max_height={state['max_pelvis_height']:.3f}", flush=True)
            episode += 1
    finally:
        runtime.close()


if __name__ == "__main__": main()
