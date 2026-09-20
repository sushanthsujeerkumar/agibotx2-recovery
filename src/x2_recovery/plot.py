"""Render saved progress without keeping an AI process or dashboard running."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("directory")
    p.add_argument("--output")
    a=p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    source=Path(a.directory)/"progress.jsonl"
    rows=[json.loads(l) for l in source.read_text().splitlines() if l.strip()]
    if not rows: raise ValueError("No training progress recorded")
    fig,axes=plt.subplots(2,1,figsize=(9,7),layout="constrained")
    for ax, key, label in zip(axes,["mean_episode_reward","success_rate"],["Episode return","Training success fraction"]):
        selected=[]
        for r in rows:
            val=r.get(key,r.get("metrics",{}).get(key))
            if val is not None: selected.append((r["environment_steps"],val))
        if selected:
            ax.plot(*zip(*selected))
        else:
            ax.text(.5,.5,"No completed-episode metric recorded",ha="center",transform=ax.transAxes)
        ax.set_ylabel(label);ax.set_xlabel("Environment steps");ax.grid(alpha=.25)
    axes[1].set_ylim(-.02,1.02)
    config_path = Path(a.directory)/"config.json"
    mode = json.loads(config_path.read_text()).get("environment", {}).get("reset_mode", "supine") if config_path.exists() else "supine"
    titles = {"balance": "X2 standing-balance PPO — standing starts", "crouch": "X2 crouch-to-standing PPO — crouch starts"}
    fig.suptitle(titles.get(mode, "X2 recovery PPO — observed training results"))
    dest=Path(a.output) if a.output else Path(a.directory)/"training_curve.png"
    dest.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(dest,dpi=150)
    print(dest)


if __name__ == "__main__": main()
