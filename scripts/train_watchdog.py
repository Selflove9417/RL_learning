#!/usr/bin/env python
"""External watchdog for PPO training runs.

WHY THIS EXISTS (Phase C failure)
---------------------------------
During the Phase C A1 pilot the launcher was written as

    cd /home/xy/IsaacTu_jump && source ./activate_jump.sh && LOG=/tmp/x.log && \
        python -u legged_gym/legged_gym/scripts/train.py ... &

The trailing `&` put the WHOLE `&&` chain into a subshell, so `LOG` was assigned
inside that subshell only. Back in the parent shell `$LOG` was empty, and the
follow-up guard

    grep -E 'NaN|Inf' "$LOG"

silently became `grep -E 'NaN|Inf' ""` -- which fails with no output and no
visible error. The run happened to be clean, so nothing was lost, but the
protection was not in effect and nobody could tell: a NaN would have gone
unnoticed, and the "stop after N KL early-stops" rule did not exist at all.

This script removes the shell-quoting class of failure entirely:

  * it takes the training command as an explicit argv tail, so no `source` or
    `&&` chain is involved and nothing depends on a shell variable surviving a
    subshell;
  * it resolves the log path to an ABSOLUTE path before starting and verifies
    the file exists and is growing, so a bad path is a hard error instead of a
    silent no-op;
  * it parses the trainer's real output format (see `parse_iteration_block`)
    rather than grepping for whole words;
  * it stops the WHOLE process group, not just the direct child, so Isaac Gym's
    worker threads cannot keep the GPU busy after the decision to stop;
  * it never deletes anything: logs and already-written checkpoints are the
    evidence, and they stay on disk after a stop;
  * it returns a distinct exit code per outcome, so a wrapper script cannot
    mistake "watchdog aborted the run" for "training completed".

USAGE
-----
    python scripts/train_watchdog.py \
        --log /abs/path/pilot.log \
        --max-early-stops 3 --max-kl-mean 0.05 \
        -- python -u legged_gym/legged_gym/scripts/train.py \
               --task myrobot_obstacle_jump --headless --max_iterations 10

Everything after `--` is the command to run. The watchdog is a supervisor, not a
launcher: it does not source a virtualenv, does not add flags, and does not
change the training configuration.

EXIT CODES
----------
    0   the training command finished on its own with exit status 0
    2   aborted: NaN or Inf in a reported metric
    3   aborted: too many PPO KL early-stops
    4   aborted: KL mean exceeded the configured ceiling
    5   aborted: the training process produced no parseable output for
        --stall-timeout seconds
    <n> the training command's own non-zero exit status (passed through
        unchanged, so a crash is never reported as success)
    64  watchdog usage error (bad arguments, unwritable log, etc.)
"""

from __future__ import print_function

import argparse
import os
import re
import signal
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Exit codes (documented in the module docstring; tests import them)
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_NAN = 2
EXIT_KL_EARLY_STOP = 3
EXIT_KL_MEAN = 4
EXIT_STALL = 5
EXIT_USAGE = 64

# Metric names exactly as `OnPolicyRunner.log()` prints them, wrapped in ANSI
# bold escapes for the iteration header. The regexes below tolerate the escapes
# and arbitrary padding because the runner right-aligns every label.
_NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
# A value slot can hold a number OR a non-finite token. `-inf` needs the sign
# INSIDE the alternative: `_NUM` cannot match it, so without this the whole
# key/value match fails and a `-inf` loss is silently skipped rather than flagged.
_VALUE = r'[-+]?(?:nan|inf)|' + _NUM
_ANSI = r'(?:\x1b\[[0-9;]*m)*'
_ANSI_RE = re.compile(_ANSI)

RE_ITERATION = re.compile(
    _ANSI + r'Learning iteration\s+(\d+)\s*/\s*(\d+)' + _ANSI)
RE_KV = re.compile(r'^\s*(?P<key>[A-Za-z][A-Za-z0-9 _/()%\-]*?):\s*'
                   r'(?P<val>' + _VALUE + r')(?:\s|$)', re.IGNORECASE)


def _to_float(token):
    """Parse one metric token; returns None for NaN/Inf/unparseable."""
    try:
        value = float(token)
    except (TypeError, ValueError):
        return None
    if value != value or value in (float('inf'), float('-inf')):
        return None
    return value


def _is_nonfinite(token):
    if token is None:
        return False
    text = str(token).strip().lower()
    if text in ('nan', 'inf', '-inf', '+inf', 'na'):
        return True
    try:
        value = float(text)
    except (TypeError, ValueError):
        return False
    return value != value or value in (float('inf'), float('-inf'))


def parse_iteration_block(lines):
    """Extract structured metrics from one `Learning iteration k/N` block.

    The trainer's block format (measured on the Phase C pilot log) is:

        ****...
                               Learning iteration 4/10
        Computation:            4480 steps/s (...)
        Value function loss:    8.7422
        Surrogate loss:         0.4715
        Mean action noise std:  0.37
        Mean reward:            2.38
        Mean episode length:    57.05
        KL (mean / p95 / max):  0.01351 / 0.04933 / 0.09248
        PPO clip fraction:      0.1354
        Policy entropy:         0.4772
        Mean grad norm:         123.6042
        Optimizer updates:      7/20 [EARLY STOPPED]
        Policy delta L2 (rel):  0.03499 (0.130%)
        Early stop reason:      KL mean (...) exceeded threshold (...) at update 8/20
        Mean episode rew_obstacle_collision: -0.0300
        Total timesteps:        ...
        Total time:             ...

    Every numeric field is optional: a block that ends early (crash, kill) still
    yields whatever was printed, and `complete` says whether the block reached its
    `Total time:` trailer.

    Returns a dict with:
        iteration, total, metrics{name: float}, nonfinite[list of (name, raw)],
        early_stopped (bool), early_stop_reason (str), kl_mean/kl_p95/kl_max,
        updates_executed, updates_planned, complete (bool)
    """
    record = {
        'iteration': None,
        'total': None,
        'metrics': {},
        'nonfinite': [],
        'early_stopped': False,
        'early_stop_reason': '',
        'kl_mean': None,
        'kl_p95': None,
        'kl_max': None,
        'updates_executed': None,
        'updates_planned': None,
        'complete': False,
    }
    combined = '\n'.join(lines)

    match = RE_ITERATION.search(combined)
    if match:
        record['iteration'] = int(match.group(1))
        record['total'] = int(match.group(2))

    if 'EARLY STOPPED' in combined.upper():
        # `Optimizer updates: 7/20 [EARLY STOPPED]` is the machine-readable
        # marker; it is printed only when the KL guard actually fired.
        record['early_stopped'] = True

    for line in lines:
        stripped = _ANSI_RE.sub('', line).rstrip('\n')

        if stripped.strip().lower().startswith('early stop reason:'):
            record['early_stop_reason'] = stripped.split(':', 1)[1].strip()
            record['early_stopped'] = True
            continue

        if 'Optimizer updates:' in stripped:
            tail = stripped.split(':', 1)[1]
            m = re.search(r'(?P<done>\d+)\s*/\s*(?P<planned>\d+)', tail)
            if m:
                record['updates_executed'] = int(m.group('done'))
                record['updates_planned'] = int(m.group('planned'))
            continue

        if 'KL (mean / p95 / max):' in stripped:
            tail = stripped.split(':', 1)[1]
            parts = re.findall(_NUM + r'|nan|NaN|inf|Inf', tail)
            for slot, name in enumerate(('kl_mean', 'kl_p95', 'kl_max')):
                if slot >= len(parts):
                    break
                token = parts[slot]
                if _is_nonfinite(token):
                    record['nonfinite'].append((name, token))
                    record['metrics'][name] = float('nan')
                else:
                    value = _to_float(token)
                    record[name] = value
                    record['metrics'][name] = value
            continue

        m = RE_KV.match(stripped)
        if not m:
            continue
        key = m.group('key').strip()
        token = m.group('val').strip()
        if _is_nonfinite(token):
            record['nonfinite'].append((key, token))
            record['metrics'][key] = float('nan')
            continue
        value = _to_float(token)
        if value is None:
            continue
        record['metrics'][key] = value
        if key == 'Value function loss':
            record['metrics']['value_loss'] = value
        elif key == 'Surrogate loss':
            record['metrics']['surrogate_loss'] = value
        elif key == 'Mean reward':
            record['metrics']['mean_reward'] = value
        elif key == 'Mean episode length':
            record['metrics']['mean_episode_length'] = value
        elif key == 'Policy entropy':
            record['metrics']['entropy'] = value
        elif key == 'Mean grad norm':
            record['metrics']['grad_norm'] = value
        elif key == 'PPO clip fraction':
            record['metrics']['clip_fraction'] = value
        elif key == 'Mean action noise std':
            record['metrics']['action_std'] = value

    if re.search(r'Total time:\s*' + _NUM, _ANSI_RE.sub('', combined)):
        record['complete'] = True
    return record


class WatchdogPolicy(object):
    """The decision rules, kept separate from process plumbing so they are testable.

    Rules (all configurable; defaults chosen for the small obstacle-jump pilot):

      NaN/Inf      -- abort IMMEDIATELY. A single non-finite metric has already
                      poisoned the optimiser state; continuing wastes GPU hours
                      and produces an unusable checkpoint.
      KL early stop-- count them. `enable_early_stopping` is the ONLY KL control
                      in this config (`schedule='fixed'`, no adaptive LR), so a
                      run that early-stops on most iterations is not learning at a
                      usable rate. Phase C hit 2/10; the default allows 3 before
                      aborting.
      KL mean      -- abort if the reported mean exceeds a hard ceiling. This is
                      the "don't quietly relax the KL limit" guard: the ceiling is
                      a watchdog parameter, NOT a change to the trainer config.
      stall        -- abort if no new iteration block appears for `stall_timeout`
                      seconds. A frozen Isaac Gym usually still burns GPU.
    """

    def __init__(self, max_early_stops=3, max_kl_mean=0.05, stall_timeout=900.0):
        self.max_early_stops = int(max_early_stops)
        self.max_kl_mean = float(max_kl_mean)
        self.stall_timeout = float(stall_timeout)
        self.early_stops = 0
        self.seen_iterations = []
        self.last_progress_time = None
        self.last_record = None

    def observe(self, record, now=None):
        """Return (exit_code, reason) when the run must stop, else (None, '')."""
        now = time.time() if now is None else now
        self.last_record = record
        if record.get('iteration') is not None:
            if not self.seen_iterations or record['iteration'] != self.seen_iterations[-1]:
                self.seen_iterations.append(record['iteration'])
            self.last_progress_time = now

        if record.get('nonfinite'):
            detail = ', '.join('%s=%s' % (k, v) for k, v in record['nonfinite'])
            return EXIT_NAN, 'non-finite metric in iteration %s: %s' % (
                record.get('iteration'), detail)

        if record.get('early_stopped'):
            self.early_stops += 1
            if self.early_stops > self.max_early_stops:
                return EXIT_KL_EARLY_STOP, (
                    'KL early stop %d/%d (limit %d); last reason: %s'
                    % (self.early_stops, self.early_stops, self.max_early_stops,
                       record.get('early_stop_reason') or 'n/a'))

        kl_mean = record.get('kl_mean')
        if kl_mean is not None and kl_mean == kl_mean and kl_mean > self.max_kl_mean:
            return EXIT_KL_MEAN, (
                'KL mean %.5f exceeded watchdog ceiling %.5f in iteration %s'
                % (kl_mean, self.max_kl_mean, record.get('iteration')))

        return None, ''

    def check_stall(self, now=None):
        """Stall rule, callable on its own so it runs even when nothing arrives."""
        now = time.time() if now is None else now
        if self.stall_timeout <= 0 or self.last_progress_time is None:
            return None, ''
        idle = now - self.last_progress_time
        if idle > self.stall_timeout:
            return EXIT_STALL, ('no new iteration for %.0f s (limit %.0f s)'
                                % (idle, self.stall_timeout))
        return None, ''


def _terminate_group(proc, grace=20.0, log=print):
    """Stop the child AND its process group, escalating gently -> firmly.

    SIGINT first: the trainer runs a plain Python loop, so it unwinds normally and
    the TensorBoard writer / diagnostics JSON are flushed. SIGTERM next. SIGKILL
    last. Nothing here removes a file: logs and checkpoints written before the
    stop must survive as evidence.
    """
    codes = []
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
        if proc.poll() is not None:
            break
        log('[watchdog] sending %s to training process group' % sig.name)
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except (OSError, ProcessLookupError):
            try:
                proc.send_signal(sig)
            except (OSError, ProcessLookupError):
                break
        try:
            proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            codes.append(sig.name)
            continue
        break
    return codes


def run_supervised(command, log_path, policy, poll_interval=0.2, log=print):
    """Run `command`, tee to `log_path`, and supervise it. Returns an exit code."""
    log_path = os.path.abspath(log_path)
    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.isdir(log_dir):
        try:
            os.makedirs(log_dir)
        except OSError as exc:
            log('[watchdog] cannot create log directory %s: %s' % (log_dir, exc))
            return EXIT_USAGE
    if os.path.exists(log_path):
        log('[watchdog] WARNING: appending to existing log %s' % log_path)

    try:
        log_fh = open(log_path, 'a', buffering=1)
    except OSError as exc:
        log('[watchdog] cannot open log file %s: %s' % (log_path, exc))
        return EXIT_USAGE
    log('[watchdog] command: %s' % ' '.join(command))
    log('[watchdog] log: %s' % log_path)
    log('[watchdog] rules: max_early_stops=%d max_kl_mean=%.4f stall=%.0fs'
        % (policy.max_early_stops, policy.max_kl_mean, policy.stall_timeout))

    # start_new_session=True gives the child its own process group so the whole
    # Isaac Gym tree can be stopped, not just the python front-end.
    proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, bufsize=1,
                            universal_newlines=True, start_new_session=True)

    policy.last_progress_time = time.time()
    block = []
    aborted = None
    # A reader thread feeds lines into `pending` so the main loop never blocks.
    # The naive form (`proc.stdout.readline()` in the main loop) blocks forever on
    # a silent child, which makes the stall rule unreachable: the watchdog would
    # wait for output it is supposed to complain about the absence of.
    try:
        import queue
        import threading
    except ImportError:                       # pragma: no cover - py2 has no queue
        queue = None
        threading = None

    if queue is not None:
        pending = queue.Queue()
        sentinel = object()

        def _reader(stream, sink):
            try:
                for raw in iter(stream.readline, ''):
                    sink.put(raw)
            finally:
                sink.put(sentinel)

        reader = threading.Thread(target=_reader, args=(proc.stdout, pending))
        reader.daemon = True
        reader.start()
    else:                                     # pragma: no cover
        pending = None
        sentinel = None
        reader = None

    eof = False
    try:
        while True:
            if pending is not None:
                try:
                    item = pending.get(timeout=poll_interval)
                except queue.Empty:
                    item = None
                if item is sentinel:
                    eof = True
                    item = None
                line = item
            else:                             # pragma: no cover
                line = proc.stdout.readline()

            if line:
                log_fh.write(line)
                sys.stdout.write(line)
                sys.stdout.flush()
                block.append(line)
                if 'Total time:' in line:
                    record = parse_iteration_block(block)
                    code, reason = policy.observe(record)
                    log('[watchdog] iteration %s: %s' % (
                        record.get('iteration'),
                        'ok' if code is None else reason))
                    if code is not None:
                        aborted = (code, reason)
                        break
                    block = []
                # Hard cap on block growth so a malformed log cannot eat memory.
                if len(block) > 400:
                    block = block[-200:]

            if eof or (line is None and proc.poll() is not None
                       and (pending is not None and pending.empty())):
                # The child closed its stdout AND has exited: nothing more can
                # arrive, so the run finished (or died) on its own.
                if proc.poll() is not None:
                    break

            # Stall rule runs every poll, independent of whether output arrived.
            code, reason = policy.check_stall(now=time.time())
            if code is not None:
                aborted = (code, reason)
                break
    finally:
        if aborted is not None:
            log_fh.flush()
            _terminate_group(proc, log=log)
        else:
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                log('[watchdog] child did not exit after EOF; stopping group')
                _terminate_group(proc, log=log)
        log_fh.flush()
        log_fh.close()

    if aborted is not None:
        code, reason = aborted
        log('[watchdog] ABORTED (exit %d): %s' % (code, reason))
        log('[watchdog] log preserved at %s' % log_path)
        return code

    returncode = proc.returncode
    if returncode is None:
        returncode = -1
    if returncode != 0:
        log('[watchdog] training exited with status %d (passed through)' % returncode)
        return returncode
    log('[watchdog] training completed normally; log at %s' % log_path)
    return EXIT_OK


def build_parser():
    parser = argparse.ArgumentParser(
        description='Supervise a PPO training run: NaN/Inf guard, KL early-stop '
                    'budget, KL ceiling, stall detection, whole-group stop.')
    parser.add_argument('--log', required=True,
                        help='absolute or relative path of the log to tee into')
    parser.add_argument('--max-early-stops', type=int, default=3,
                        help='abort after this many PPO KL early stops (default 3)')
    parser.add_argument('--max-kl-mean', type=float, default=0.05,
                        help='abort if a reported KL mean exceeds this (default 0.05)')
    parser.add_argument('--stall-timeout', type=float, default=900.0,
                        help='abort after this many seconds without an iteration '
                             'block (default 900; 0 disables)')
    parser.add_argument('--poll-interval', type=float, default=0.2)
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='-- followed by the training command')
    return parser


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    parser = build_parser()
    args = parser.parse_args(argv)
    command = list(args.command)
    while command and command[0] == '--':
        command.pop(0)
    if not command:
        parser.print_usage(sys.stderr)
        print('error: no training command given after --', file=sys.stderr)
        return EXIT_USAGE
    policy = WatchdogPolicy(max_early_stops=args.max_early_stops,
                            max_kl_mean=args.max_kl_mean,
                            stall_timeout=args.stall_timeout)
    return run_supervised(command, args.log, policy,
                          poll_interval=args.poll_interval)


if __name__ == '__main__':
    sys.exit(main())
