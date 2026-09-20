#!/usr/bin/env python3
"""Check the submitted model files and saved evidence without starting simulation."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
folder = root/'artifacts/submission/final_recovery'
manifest = json.loads((folder/'manifest.json').read_text())
for name, expected in manifest['files_sha256'].items():
    actual = hashlib.sha256((folder/name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Checksum mismatch: {name}')
for name in ['evaluation_5.json', 'evaluation_10.json']:
    report = json.loads((folder/name).read_text())
    if not (report['passes'] == report['episodes'] == len(report['results'])
            and all(row['passed'] and row['limits']['ok'] for row in report['results'])):
        raise SystemExit(f'Saved evaluation failed: {name}')
for name in ['success.json', 'wall_timeout.json', 'sim_timeout.json']:
    report = json.loads((folder/'ros'/name).read_text())
    if not report['passed']:
        raise SystemExit(f'Saved ROS check failed: {name}')
print('PASS: frozen model checksums and saved evaluation/ROS evidence. This does not rerun physics.')
