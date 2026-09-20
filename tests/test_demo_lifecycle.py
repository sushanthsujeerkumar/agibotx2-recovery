"""Interactive demo lifetime, reset and failure behavior without a GUI dependency."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def demo(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / 'scripts/demo_full_recovery.py'
    spec = importlib.util.spec_from_file_location('demo_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.time, 'sleep', lambda _: None)
    monkeypatch.setattr(module.time, 'monotonic', lambda: 0.)
    return module, tmp_path / 'result.json'


def run_case(demo, monkeypatch, *, flags=(), close_at=900, restart_at=None, fault_at=None):
    module, output = demo
    instances = []

    class Runtime:
        def __init__(self, *args, key_callback=None, **kwargs):
            self.callback = key_callback
            self.total_steps = self.steps = self.polls = self.resets = 0
            self.closed = False
            self.clean_hold_time = 0.
            self.limits = SimpleNamespace(ok=True)
            self.viewer = SimpleNamespace(is_running=self.running, sync=lambda **kw: None)
            instances.append(self)

        def running(self):
            self.polls += 1
            if self.polls == restart_at:
                self.callback(ord('R'))
            return self.polls <= close_at

        def snapshot(self, success, invalid):
            return {'sim_time': self.steps * .02, 'invalid': invalid}

        def step(self):
            self.steps += 1
            self.total_steps += 1
            self.clean_hold_time = max(0., self.steps * .02 - 12.86)
            if self.total_steps == fault_at:
                self.limits.ok = False
            return self.snapshot(False, False)

        def reset(self, seed):
            self.resets += 1
            self.steps = 0
            self.clean_hold_time = 0.
            self.limits.ok = True
            return self.snapshot(False, False)

        def close(self):
            self.closed = True

    monkeypatch.setattr(module, 'RecoveryRuntime', Runtime)
    monkeypatch.setattr('sys.argv', ['demo', '--render', '--output', str(output), *flags])
    status = module.main()
    runtime = instances[0]
    assert runtime.closed
    return status, runtime, json.loads(output.read_text())


def test_keep_open_continues_physics_and_preserves_15s_assessment(demo, monkeypatch):
    status, runtime, report = run_case(demo, monkeypatch, flags=['--keep-open'])
    assert status == 0 and runtime.total_steps == 900
    assert report['sim_time'] == 18.
    assert report['assessment_15s'] == {'sim_time': 15., 'invalid': False, 'passed': True}
    assert report['end_reason'] == 'window_closed'


def test_once_overrides_launcher_keep_open(demo, monkeypatch):
    status, runtime, report = run_case(demo, monkeypatch, flags=['--keep-open', '--once'])
    assert status == 0 and runtime.total_steps == 750
    assert report['sim_time'] == 15.


def test_close_before_first_step_does_not_claim_success(demo, monkeypatch):
    status, runtime, report = run_case(demo, monkeypatch, flags=['--keep-open'], close_at=0)
    assert status == 1 and runtime.total_steps == 0
    assert report['assessment_15s'] is None and not report['passed']


def test_restart_runs_new_episode_in_same_window(demo, monkeypatch):
    status, runtime, report = run_case(demo, monkeypatch, flags=['--keep-open'],
                                       restart_at=801, close_at=1600)
    assert status == 0 and runtime.resets == 1
    assert runtime.total_steps == 1600 and runtime.steps == 800
    assert report['sim_time'] == 16. and report['assessment_15s']['passed']


def test_fault_pauses_physics_until_close(demo, monkeypatch):
    status, runtime, report = run_case(demo, monkeypatch, flags=['--keep-open'], fault_at=5,
                                       close_at=20)
    assert status == 1 and runtime.total_steps == 5
    assert not report['passed'] and not runtime.limits.ok


def test_restart_recovers_from_latched_fault(demo, monkeypatch):
    status, runtime, report = run_case(demo, monkeypatch, flags=['--keep-open'], fault_at=5,
                                       restart_at=10, close_at=800)
    assert status == 0 and runtime.resets == 1 and runtime.limits.ok
    assert report['assessment_15s']['passed']


def test_headless_keep_open_rejected(demo, monkeypatch):
    module, _ = demo
    monkeypatch.setattr('sys.argv', ['demo', '--keep-open'])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
