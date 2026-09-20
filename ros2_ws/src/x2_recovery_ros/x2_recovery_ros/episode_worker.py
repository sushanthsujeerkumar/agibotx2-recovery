"""Simulator subprocess. This module deliberately has no rclpy dependency."""

import math
import time
import traceback


def checked_frame(frame):
    """Reject invalid simulator data rather than publish fabricated joint states."""
    names = list(frame['joint_names'])
    positions = [float(value) for value in frame['joint_positions']]
    sim_time = float(frame['sim_time'])
    if not names or len(names) != len(positions):
        raise ValueError('Simulator joint names and positions must have equal nonzero lengths')
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError('Simulator joint names must be unique nonempty strings')
    if not all(math.isfinite(p) for p in positions):
        raise ValueError('Simulator joint positions contain a non-finite value')
    if not math.isfinite(sim_time) or sim_time < 0.0:
        raise ValueError('Simulator timestamp must be finite and nonnegative')
    return {
        'joint_names': names,
        'joint_positions': positions,
        'sim_time': sim_time,
        'success': bool(frame['success']),
        'invalid': bool(frame['invalid']),
    }


def run_episode(config, connection):
    """Execute one actual runtime episode; send frames and a terminal result."""
    runtime = None
    try:
        from x2_recovery.runtime import RecoveryRuntime

        runtime = RecoveryRuntime(
            controller=config['controller'],
            checkpoint=config['checkpoint'] or None,
            render=config['render'],
            seed=config['seed'],
        )
        runtime.reset(seed=config['seed'])
        control_dt = float(runtime.control_dt)
        if not math.isfinite(control_dt) or control_dt <= 0.0:
            raise ValueError('RecoveryRuntime.control_dt must be finite and positive')
        next_step_at = time.monotonic()
        previous_sim_time = None
        elapsed_sim_time = 0.0
        while True:
            raw = runtime.step()
            frame = checked_frame(raw)
            if previous_sim_time is not None:
                delta = frame['sim_time'] - previous_sim_time
                if delta <= 0.0:
                    raise ValueError('Simulator time did not advance')
                elapsed_sim_time += delta
            else:
                elapsed_sim_time = control_dt
            previous_sim_time = frame['sim_time']
            connection.send(('frame', frame))
            if frame['invalid']:
                connection.send(('result', {'success': False, 'reason': 'invalid simulator state'}))
                break
            limits = raw.get('trajectory_limits')
            if not isinstance(limits, dict) or not limits.get('ok', False):
                connection.send(('result', {'success': False, 'reason': 'trajectory limits failed or unavailable'}))
                break
            if frame['success']:
                connection.send(('result', {'success': True, 'reason': 'stable standing success checks passed'}))
                break
            if elapsed_sim_time >= config['max_sim_duration_s']:
                connection.send(('result', {'success': False, 'reason': 'simulation duration timeout'}))
                break
            if config['realtime']:
                next_step_at += control_dt
                delay = next_step_at - time.monotonic()
                if delay > 0.0:
                    time.sleep(delay)
    except BaseException as error:
        try:
            connection.send(('result', {
                'success': False,
                'reason': f'{type(error).__name__}: {error}',
                'traceback': traceback.format_exc(),
            }))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        if runtime is not None:
            try:
                runtime.close()
            except Exception:
                pass
        connection.close()
