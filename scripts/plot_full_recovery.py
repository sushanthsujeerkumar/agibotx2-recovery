#!/usr/bin/env python3
"""Plot recorded PPO episode return without treating reward as success rate."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run', type=Path, default=Path('artifacts/submission/final_recovery'))
args = parser.parse_args()
rows = [json.loads(line) for line in (args.run/'progress.jsonl').read_text().splitlines()]
rows = [row for row in rows if row.get('mean_episode_reward') is not None]
x = np.array([row['environment_steps']/1e6 for row in rows])
y = np.array([row['mean_episode_reward'] for row in rows])
fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
ax.plot(x, y, color='#bdd0df', linewidth=.8, label='Logged mean episode return')
window = min(50, len(y))
ax.plot(x[window-1:], np.convolve(y, np.ones(window)/window, mode='valid'),
        color='#245875', linewidth=1.7, label=f'{window}-update rolling mean')
ax.set(xlabel='Cumulative environment transitions (millions)', ylabel='Episode return',
       title='Full recovery: PPO feedback on a frozen motion prior')
ax.grid(alpha=.2); ax.legend(frameon=False)
fig.savefig(args.run/'training_curve.png', dpi=170)
plt.close(fig)
