"""Short measured throughput test before selecting a training batch size."""
import argparse
import gc
import json
import time
from pathlib import Path
import torch
from .env import X2RecoveryEnv


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--sizes",nargs="+",type=int,default=[128,256,512])
    p.add_argument("--steps",type=int,default=50)
    p.add_argument("--output",default="artifacts/benchmark.json")
    a=p.parse_args()
    results=[]
    for n in a.sizes:
        start=time.monotonic()
        env=X2RecoveryEnv(n)
        setup=time.monotonic()-start
        actions=torch.zeros((n,env.num_actions),device="cuda")
        for _ in range(5): env.step(actions)
        torch.cuda.synchronize()
        start=time.monotonic()
        for _ in range(a.steps): env.step(actions)
        torch.cuda.synchronize()
        elapsed=time.monotonic()-start
        result={"num_envs":n,"setup_seconds":setup,"measured_seconds":elapsed,
                "control_steps_per_second":n*a.steps/elapsed,
                "torch_peak_allocated_mib":torch.cuda.max_memory_allocated()/2**20,
                "note":"zero-action physics throughput, excludes PPO updates; Warp allocations not included in torch memory"}
        print(json.dumps(result),flush=True);results.append(result)
        env.close();del env,actions;gc.collect();torch.cuda.empty_cache()
    dest=Path(a.output);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(results,indent=2)+"\n")


if __name__ == "__main__": main()
