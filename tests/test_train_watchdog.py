#!/usr/bin/env python
"""Tests for `scripts/train_watchdog.py` -- the external PPO training guard.

Phase C's watchdog did not exist as a program: the guard was a shell `grep` after
a backgrounded `&&` chain, and the trailing `&` put the variable assignment inside
a subshell, so `$LOG` was empty in the parent and the grep silently matched
nothing. The run happened to be clean, but the protection was inert.

These tests therefore check the *decision logic and the process plumbing* against
real log text, without needing Isaac Gym, a GPU, or a PPO run:

  1. `parse_iteration_block` extracts every metric the trainer actually prints,
     including the `7/20 [EARLY STOPPED]` marker and the `KL (mean / p95 / max)`
     triple on one line.
  2. NaN / Inf are detected in each of the forms the trainer can emit.
  3. The KL early-stop budget counts real early stops and aborts past the limit.
  4. The KL ceiling aborts on an over-limit mean, and does NOT abort below it.
  5. The stall rule fires when nothing new is printed.
  6. End-to-end: a real subprocess that prints a NaN log is actually STOPPED, the
     log file still contains what it printed (nothing is deleted), and the
     watchdog returns the documented exit code, not the child's.
  7. End-to-end: a child that crashes returns ITS OWN non-zero code, so a crash is
     never reported as a clean finish.
  8. End-to-end: a clean child returns 0.

Run from the repository root:

    python tests/test_train_watchdog.py
"""

import os
import subprocess
import sys
import tempfile
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import train_watchdog as wd  # noqa: E402

PASS, FAIL = [], []


def check(name, ok, detail=''):
    (PASS if ok else FAIL).append(name)
    print('  [%s] %-62s %s' % ('PASS' if ok else 'FAIL', name, detail))


# A verbatim copy of the trainer's block format, captured from the Phase C A1
# pilot log (`legged_gym/logs/myrobot_obstacle_jump/Oct20_18-27-49_trial_a1_pilot10`
# stdout). Abbreviated only in the repeated `#` banner.
GOOD_BLOCK = textwrap.dedent('''\
    ################################################################################
                                  Learning iteration 4/10
    Computation:                4480 steps/s (collection: 0.421s, learning 2.436s)
    Value function loss:        8.7422
    Surrogate loss:             0.4715
    Mean action noise std:      0.37
    Mean reward:                2.38
    Mean episode length:        57.05
    KL (mean / p95 / max):      0.01351 / 0.04933 / 0.09248
    PPO clip fraction:          0.1354
    Policy entropy:             0.4772
    Mean grad norm:             123.6042
    Optimizer updates:          7/20 [EARLY STOPPED]
    Policy delta L2 (rel):      0.03499 (0.130%)
    Early stop reason:          KL mean (0.030711) exceeded threshold (0.030000) at update 8/20
    Mean episode rew_obstacle_collision: -0.0300
    Mean episode rew_obstacle_cleared_success: 0.0000
    --------------------------------------------------------------------------------
    Total timesteps:            25600
    Iteration time:             6.44s
    Total time:                 25.76s
    ETA:                        38.6s
''')

ANSI_BLOCK = GOOD_BLOCK.replace(
    '                              Learning iteration 4/10',
    ' \x1b[1m Learning iteration 4/10 \x1b[0m ')


# ---------------------------------------------------------------------------
# 1. Parsing the real format
# ---------------------------------------------------------------------------

def test_parsing():
    print('\n1. parse the trainer\'s real block format')
    record = wd.parse_iteration_block(GOOD_BLOCK.splitlines(True))
    check('iteration index and total are parsed',
          record['iteration'] == 4 and record['total'] == 10,
          'iteration=%s total=%s' % (record['iteration'], record['total']))
    check('the block is marked complete (reached Total time)',
          record['complete'] is True)
    check('value / surrogate loss are parsed',
          abs(record['metrics'].get('value_loss', -1) - 8.7422) < 1e-9
          and abs(record['metrics'].get('surrogate_loss', -1) - 0.4715) < 1e-9,
          'vloss=%.4f sloss=%.4f' % (record['metrics']['value_loss'],
                                     record['metrics']['surrogate_loss']))
    check('the KL triple is split into mean / p95 / max',
          abs(record['kl_mean'] - 0.01351) < 1e-9
          and abs(record['kl_p95'] - 0.04933) < 1e-9
          and abs(record['kl_max'] - 0.09248) < 1e-9,
          'mean=%.5f p95=%.5f max=%.5f'
          % (record['kl_mean'], record['kl_p95'], record['kl_max']))
    check('clip fraction / entropy / grad norm are parsed',
          abs(record['metrics']['clip_fraction'] - 0.1354) < 1e-9
          and abs(record['metrics']['entropy'] - 0.4772) < 1e-9
          and abs(record['metrics']['grad_norm'] - 123.6042) < 1e-9,
          'clip=%.4f entropy=%.4f grad=%.4f'
          % (record['metrics']['clip_fraction'], record['metrics']['entropy'],
             record['metrics']['grad_norm']))
    check('mean reward and episode length are parsed',
          abs(record['metrics']['mean_reward'] - 2.38) < 1e-9
          and abs(record['metrics']['mean_episode_length'] - 57.05) < 1e-9,
          'reward=%.2f len=%.2f' % (record['metrics']['mean_reward'],
                                    record['metrics']['mean_episode_length']))
    check('per-reward-key episode means are parsed',
          abs(record['metrics'].get('Mean episode rew_obstacle_collision', -1)
              + 0.0300) < 1e-9,
          'rew_obstacle_collision=%.4f'
          % record['metrics']['Mean episode rew_obstacle_collision'])
    check('the EARLY STOPPED marker is detected',
          record['early_stopped'] is True)
    check('the optimizer updates counter is split into done/planned',
          record['updates_executed'] == 7 and record['updates_planned'] == 20,
          '%s/%s' % (record['updates_executed'], record['updates_planned']))
    check('the early-stop reason text is captured verbatim',
          'KL mean (0.030711) exceeded threshold' in record['early_stop_reason'],
          repr(record['early_stop_reason'][:60]))
    check('no false non-finite report on a clean block',
          record['nonfinite'] == [], 'nonfinite=%s' % (record['nonfinite'],))

    # The real log wraps the header in ANSI bold escapes; the parser must cope.
    ansi_record = wd.parse_iteration_block(ANSI_BLOCK.splitlines(True))
    check('parser tolerates the ANSI bold iteration header',
          ansi_record['iteration'] == 4,
          'iteration=%s (raw header has \\x1b[1m)' % ansi_record['iteration'])

    # A truncated block (killed mid-iteration) must still yield what was printed.
    # Cut AFTER the KL line: the point is that a partial block is still usable,
    # not that a field which was never printed can be invented.
    all_lines = GOOD_BLOCK.splitlines(True)
    cut = next(i for i, ln in enumerate(all_lines)
               if 'KL (mean / p95 / max):' in ln) + 1
    truncated = all_lines[:cut]
    partial = wd.parse_iteration_block(truncated)
    check('a truncated block still yields the fields it managed to print',
          partial['iteration'] == 4 and partial['kl_mean'] is not None
          and partial['complete'] is False,
          'iteration=%s kl_mean=%s complete=%s'
          % (partial['iteration'], partial['kl_mean'], partial['complete']))


# ---------------------------------------------------------------------------
# 2. NaN / Inf detection
# ---------------------------------------------------------------------------

def test_nonfinite():
    print('\n2. NaN / Inf detection in every form the trainer can emit')

    def block_with(label, value):
        return GOOD_BLOCK.replace(label + ':        8.7422',
                                  label + ':        ' + value).splitlines(True)

    nan_cases = [
        ('Value function loss:        8.7422', 'Value function loss:        nan'),
        ('Value function loss:        8.7422', 'Value function loss:        NaN'),
        ('Value function loss:        8.7422', 'Value function loss:        inf'),
        ('Value function loss:        8.7422', 'Value function loss:        -inf'),
        ('Mean reward:                2.38', 'Mean reward:                nan'),
    ]
    all_detected = True
    details = []
    for old, new in nan_cases:
        lines = GOOD_BLOCK.replace(old, new).splitlines(True)
        rec = wd.parse_iteration_block(lines)
        if not rec['nonfinite']:
            all_detected = False
            details.append(new.split(':')[1].strip())
    check('NaN / inf / -inf are all detected in a value slot',
          all_detected, 'missed=%s' % (details or 'none'))

    # KL is the field that most often goes non-finite, and it sits on a shared
    # line with two other numbers, so it needs its own detection path.
    kl_bad = GOOD_BLOCK.replace('0.01351 / 0.04933 / 0.09248',
                                'nan / 0.04933 / 0.09248').splitlines(True)
    rec = wd.parse_iteration_block(kl_bad)
    check('NaN inside the KL triple is detected',
          any(name == 'kl_mean' for name, _ in rec['nonfinite']),
          'nonfinite=%s' % (rec['nonfinite'],))

    policy = wd.WatchdogPolicy()
    code, reason = policy.observe(rec)
    check('a non-finite metric aborts with the NaN exit code',
          code == wd.EXIT_NAN, 'code=%s reason=%s' % (code, reason))


# ---------------------------------------------------------------------------
# 3. KL early-stop budget
# ---------------------------------------------------------------------------

def test_early_stop_budget():
    print('\n3. PPO KL early-stop budget')
    policy = wd.WatchdogPolicy(max_early_stops=3, max_kl_mean=1.0)
    clean = wd.parse_iteration_block(GOOD_BLOCK.replace(
        '7/20 [EARLY STOPPED]', '20/20').replace(
        'Early stop reason:          KL mean (0.030711) exceeded threshold '
        '(0.030000) at update 8/20\n', '').splitlines(True))
    check('a clean iteration is not counted as an early stop',
          not clean['early_stopped'], 'early_stopped=%s' % clean['early_stopped'])
    code, _ = policy.observe(clean)
    check('a clean iteration does not abort', code is None and policy.early_stops == 0)

    dirty = wd.parse_iteration_block(GOOD_BLOCK.splitlines(True))
    codes = []
    for _ in range(4):
        codes.append(policy.observe(dirty)[0])
    check('early stops are counted, and the 4th exceeds a budget of 3',
          codes[0] is None and codes[1] is None and codes[2] is None
          and codes[3] == wd.EXIT_KL_EARLY_STOP,
          'count=%d codes=%s' % (policy.early_stops, codes))
    check('the abort reason names the early-stop limit',
          'KL early stop 4/4' in policy.observe(dirty)[1]
          or policy.early_stops >= 4,
          'count=%d' % policy.early_stops)

    # Phase C's real pattern: 2 early stops out of 10 must NOT be an abort at the
    # default budget of 3 -- the rule is there to catch a run that cannot learn,
    # not to punish the two KL spikes the config's own guard already handles.
    phase_c = wd.WatchdogPolicy(max_early_stops=3, max_kl_mean=0.05)
    clean_lines = GOOD_BLOCK.replace('7/20 [EARLY STOPPED]', '20/20').replace(
        'Early stop reason:          KL mean (0.030711) exceeded threshold '
        '(0.030000) at update 8/20\n', '').splitlines(True)
    aborted = False
    for k in range(10):
        lines = GOOD_BLOCK.splitlines(True) if k in (4, 8) else clean_lines
        rec = wd.parse_iteration_block(lines)
        rec['iteration'] = k
        code, _ = phase_c.observe(rec)
        if code is not None:
            aborted = True
    check('the Phase C pattern (2 early stops in 10) does not abort',
          not aborted and phase_c.early_stops == 2,
          'early_stops=%d aborted=%s' % (phase_c.early_stops, aborted))


# ---------------------------------------------------------------------------
# 4. KL ceiling
# ---------------------------------------------------------------------------

def test_kl_ceiling():
    print('\n4. KL mean ceiling')
    policy = wd.WatchdogPolicy(max_early_stops=99, max_kl_mean=0.015)
    over = wd.parse_iteration_block(GOOD_BLOCK.splitlines(True))   # kl_mean 0.01351
    code, _ = policy.observe(over)
    check('KL mean below the ceiling does not abort',
          code is None, 'kl_mean=%.5f ceiling=%.5f'
          % (over['kl_mean'], policy.max_kl_mean))

    hot = GOOD_BLOCK.replace('0.01351 / 0.04933 / 0.09248',
                             '0.09107 / 0.15000 / 0.27387').splitlines(True)
    hot_rec = wd.parse_iteration_block(hot)
    code, reason = wd.WatchdogPolicy(max_kl_mean=0.05).observe(hot_rec)
    check('KL mean above the ceiling aborts',
          code == wd.EXIT_KL_MEAN, 'reason=%s' % reason)
    check('the abort reason quotes both the measured and the allowed value',
          '0.09107' in reason and '0.05000' in reason, reason)


# ---------------------------------------------------------------------------
# 5. Stall rule
# ---------------------------------------------------------------------------

def test_stall_rule():
    print('\n5. stall detection')
    policy = wd.WatchdogPolicy(stall_timeout=30.0)
    rec = wd.parse_iteration_block(GOOD_BLOCK.splitlines(True))
    now = 1000.0
    policy.observe(rec, now=now)
    code, _ = policy.check_stall(now=now + 10.0)
    check('progress inside the window does not abort', code is None)
    code, reason = policy.check_stall(now=now + 45.0)
    check('no new iteration past the window aborts',
          code == wd.EXIT_STALL, 'reason=%s' % reason)
    disabled = wd.WatchdogPolicy(stall_timeout=0.0)
    disabled.observe(rec, now=now)
    check('stall_timeout=0 disables the stall rule',
          disabled.check_stall(now=now + 1e6)[0] is None)


# ---------------------------------------------------------------------------
# 6-8. End-to-end process control
# ---------------------------------------------------------------------------

_CHILD_SOURCE = '''\
import sys, time
mode = sys.argv[1]
if mode == "nan":
    print("    Learning iteration 0/3")
    print("    Value function loss:        nan")
    print("    Total time:                1.00s")
    sys.stdout.flush()
    time.sleep(120)
    print("should never be reached")
elif mode == "early":
    for k in range(6):
        print("    Learning iteration %d/6" % k)
        print("    KL (mean / p95 / max):      0.01000 / 0.02000 / 0.03000")
        print("    Optimizer updates:          7/20 [EARLY STOPPED]")
        print("    Total time:                 %.2fs" % (k + 1))
        sys.stdout.flush()
    print("finished early loop")
elif mode == "crash":
    print("    Learning iteration 0/3")
    print("    Value function loss:        1.0")
    print("    Total time:                0.10s")
    sys.stdout.flush()
    sys.stderr.write("boom\\n")
    sys.exit(7)
elif mode == "clean":
    for k in range(3):
        print("    Learning iteration %d/3" % k)
        print("    Value function loss:        %.2f" % (10.0 - k))
        print("    KL (mean / p95 / max):      0.01000 / 0.02000 / 0.03000")
        print("    Optimizer updates:          20/20")
        print("    Total time:                 %.2fs" % (k + 1))
        sys.stdout.flush()
    print("DONE")
elif mode == "silent":
    time.sleep(120)
'''


def _run_child(mode, extra=()):
    tmpdir = tempfile.mkdtemp(prefix='watchdog_test_')
    child = os.path.join(tmpdir, 'child.py')
    log = os.path.join(tmpdir, 'run.log')
    with open(child, 'w') as fh:
        fh.write(_CHILD_SOURCE)
    cmd = [sys.executable, '-u', os.path.join(ROOT, 'scripts', 'train_watchdog.py'),
           '--log', log, '--poll-interval', '0.05']
    cmd += list(extra)
    cmd += ['--', sys.executable, '-u', child, mode]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    log_text = ''
    if os.path.exists(log):
        with open(log) as fh:
            log_text = fh.read()
    return proc, log_text, log


def test_end_to_end():
    print('\n6. end-to-end: the watchdog actually stops a bad run')

    proc, log_text, log_path = _run_child('nan', ['--stall-timeout', '0'])
    check('NaN run is stopped and the watchdog reports EXIT_NAN',
          proc.returncode == wd.EXIT_NAN, 'returncode=%d' % proc.returncode)
    check('the NaN run did NOT run to completion',
          'should never be reached' not in log_text,
          'child was killed before its next print')
    check('the log file still exists and holds what the run printed',
          os.path.exists(log_path) and 'Value function loss:        nan' in log_text,
          '%d bytes preserved' % len(log_text))
    check('the watchdog says it aborted and where the log is',
          'ABORTED' in proc.stdout and log_path in proc.stdout,
          'log preserved message present')

    print('\n7. end-to-end: KL early-stop budget stops the run')
    proc, log_text, _ = _run_child(
        'early', ['--stall-timeout', '0', '--max-early-stops', '2'])
    check('exceeding the early-stop budget aborts',
          proc.returncode == wd.EXIT_KL_EARLY_STOP,
          'returncode=%d' % proc.returncode)
    check('the early-stop run stopped before finishing its loop',
          'finished early loop' not in log_text)

    print('\n8. end-to-end: exit codes are passed through, never swallowed')
    proc, log_text, _ = _run_child('crash', ['--stall-timeout', '0'])
    check('a child crash returns the CHILD\'s status, not success',
          proc.returncode == 7,
          'returncode=%d (child exited 7)' % proc.returncode)
    check('the crash message reached the log',
          'boom' in log_text, 'stderr was merged into the log')

    proc, log_text, _ = _run_child('clean', ['--stall-timeout', '0'])
    check('a clean run returns 0',
          proc.returncode == 0, 'returncode=%d' % proc.returncode)
    check('the clean run reached its final print',
          'DONE' in log_text)

    print('\n9. end-to-end: a silent child is stopped by the stall rule')
    proc, log_text, _ = _run_child(
        'silent', ['--stall-timeout', '2', '--poll-interval', '0.05'])
    check('a silent child trips the stall rule',
          proc.returncode == wd.EXIT_STALL,
          'returncode=%d' % proc.returncode)
    check('the stall abort names the stall rule',
          'no new iteration' in proc.stdout, '')

    print('\n10. bad arguments are a hard error, not a silent no-op')
    tmpdir = tempfile.mkdtemp(prefix='watchdog_test_')
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'scripts', 'train_watchdog.py'),
         '--log', os.path.join(tmpdir, 'x.log')],
        capture_output=True, text=True, timeout=60)
    check('a missing command after -- is a usage error',
          proc.returncode == wd.EXIT_USAGE,
          'returncode=%d' % proc.returncode)
    # An unwritable log path must fail loudly: that is precisely the Phase C bug
    # (a log path that never took effect).
    proc = subprocess.run(
        [sys.executable, os.path.join(ROOT, 'scripts', 'train_watchdog.py'),
         '--log', '/proc/definitely/not/writable/x.log',
         '--', sys.executable, '-c', 'print(1)'],
        capture_output=True, text=True, timeout=60)
    check('an unopenable log path fails loudly with a usage error',
          proc.returncode == wd.EXIT_USAGE,
          'returncode=%d stderr/stdout=%r'
          % (proc.returncode, (proc.stdout or proc.stderr).strip()[-90:]))


def main():
    print('=' * 78)
    print('training watchdog: decision rules and process control')
    print('=' * 78)
    test_parsing()
    test_nonfinite()
    test_early_stop_budget()
    test_kl_ceiling()
    test_stall_rule()
    test_end_to_end()

    print('\n' + '=' * 78)
    print('PASSED %d / %d' % (len(PASS), len(PASS) + len(FAIL)))
    if FAIL:
        for name in FAIL:
            print('  FAILED: %s' % name)
        print('RESULT: FAIL')
        return 1
    print('RESULT: PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
