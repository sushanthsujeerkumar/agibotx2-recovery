#!/usr/bin/env python3
"""Wait cheaply for training, then preserve a checkpoint and real five-episode evidence."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="artifacts/runs/local")
    p.add_argument("--output", default="artifacts/submission/local_initial")
    p.add_argument("--video", action="store_true")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = (root/a.run_dir).resolve()
    dest = (root/a.output).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    status_path = dest/"finalization.json"
    def status(state, **more):
        tmp=status_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({"state":state,"updated_unix":time.time(),**more},indent=2))
        tmp.replace(status_path)
    if status_path.exists() and json.loads(status_path.read_text()).get('state') == 'COMPLETED':
        print('Finalization already completed; preserving existing evidence.');return
    status("WAITING_FOR_TRAINING")
    try:
        while True:
            current=json.loads((run/"status.json").read_text())
            if current["state"] in {"PAUSED","COMPLETED","FAILED"}: break
            proc=Path(f"/proc/{current['pid']}/cmdline")
            if not proc.exists() or b"x2_recovery.train" not in proc.read_bytes():
                current=json.loads((run/"status.json").read_text())
                if current["state"] in {"PAUSED","COMPLETED","FAILED"}: break
                raise RuntimeError("Training process ended without a final status; inspect console.log")
            time.sleep(60)
        if current["state"] == "FAILED":
            raise RuntimeError(f"Training failed: {current.get('error','see console.log')}")
        status("EVALUATING",training_state=current)
        meta=json.loads((run/"actor_latest.json").read_text())
        actor=Path(meta["actor"])
        checkpoint=Path(meta["checkpoint"])
        shutil.copy2(actor,dest/"actor.pt")
        shutil.copy2(checkpoint,dest/"checkpoint.pt")
        for name in ["config.json","progress.jsonl","status.json"]:
            shutil.copy2(run/name,dest/name)
        command=[sys.executable,"-m","x2_recovery.evaluate","--controller","policy",
                 "--checkpoint",str(dest/"actor.pt"),"--episodes","5","--seed","1001",
                 "--output",str(dest/"evaluation")]
        if a.video: command.append("--video")
        with (dest/"evaluation.log").open('w') as log:
            subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,"-m","x2_recovery.plot",str(dest),"--output",str(dest/"training_curve.png")],cwd=root,check=True)
        summary=json.loads((dest/"evaluation/summary.json").read_text())
        manifest={"run_directory":str(run),"iteration":meta["iteration"],
                  "environment_steps":meta["environment_steps"],"training_wall_time_s":current.get('wall_time_s'),
                  "actor_sha256":hashlib.sha256((dest/"actor.pt").read_bytes()).hexdigest(),
                  "checkpoint_sha256":hashlib.sha256((dest/"checkpoint.pt").read_bytes()).hexdigest(),
                  "successes":summary["successes"],"episodes":summary["episodes"],
                  "controller":"learned_policy","evaluation_command":command}
        (dest/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        status("COMPLETED",**manifest)
        print(json.dumps(manifest),flush=True)
    except Exception as exc:
        status("FAILED",error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == '__main__': main()
