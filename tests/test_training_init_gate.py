"""Startup admission decisions, including real CLI exit codes; no Gym/GPU needed."""

import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

from training_init_gate import evaluate


def clean_evidence():
    root = [0., 0., 0.4, 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]
    frames = []
    for step in range(25):
        frames.append({
            'step': step, 'storage_step': step + 1, 'done_env_ids': [],
            'leg_only_env_ids': [], 'supported_env_ids': list(range(64)) if step >= 4 else [],
            'root_states': [root[:] for _ in range(64)],
            'observations': [[0.] * 35 for _ in range(64)],
            'actions': [[0.] * 6 for _ in range(64)],
            'rewards': [0.] * 64, 'stored_rewards': [0.] * 64,
            'actor_loss_mask': [0.] * 64, 'stored_actor_masks': [0.] * 64,
            'time_outs': [0.] * 64, 'wheel_normal_forces': [[100., 100.]] * 64,
            'leg_loaded': [False] * 64, 'terminal': [],
        })
    return {'pipeline': 'gpu', 'num_envs': 64,
            'initial_state': {'root_states': [root[:] for _ in range(64)]},
            'frames': frames, 'policy_weights_unchanged': True,
            'optimizer_steps_before': 0, 'optimizer_steps_after': 0,
            'reset_calls': 1, 'validation_errors': []}


class AdmissionTests(unittest.TestCase):
    def test_normal_settling_is_allowed(self):
        self.assertEqual(evaluate(clean_evidence())['status'], 'PASS')

    def test_early_done_fails_even_after_later_recovery(self):
        data = clean_evidence()
        data['frames'][13]['done_env_ids'] = [0]
        verdict = evaluate(data)
        self.assertEqual(verdict['status'], 'FAIL')
        self.assertEqual(verdict['terminated_env_ids'], [0])

    def test_leg_only_counter_requires_consecutive_frames(self):
        for steps, expected in (([4, 5], 'PASS'), ([4, 6, 8], 'PASS'), ([4, 5, 6], 'FAIL')):
            data = clean_evidence()
            for step in steps:
                data['frames'][step]['leg_only_env_ids'] = [0]
            with self.subTest(steps=steps):
                self.assertEqual(evaluate(data)['status'], expected)

    def test_checks_every_environment(self):
        data = clean_evidence()
        for frame in data['frames']:
            frame['supported_env_ids'] = list(range(63))
        verdict = evaluate(data)
        self.assertEqual(verdict['status'], 'FAIL')
        self.assertEqual(verdict['unsupported_env_ids'], [63])

    def test_validation_and_mutation_failures(self):
        for key, value in (('validation_errors', ['observations nonfinite at 1']),
                           ('validation_errors', ['actor mask storage mismatch at 2']),
                           ('policy_weights_unchanged', False),
                           ('optimizer_steps_after', 1), ('reset_calls', 2),
                           ('pipeline', 'cpu'), ('num_envs', 63)):
            data = clean_evidence()
            data[key] = value
            with self.subTest(key=key, value=value):
                self.assertEqual(evaluate(data)['status'], 'FAIL')

    def test_missing_evidence_is_incomplete(self):
        complete = clean_evidence()
        missing = copy.deepcopy(complete)
        del missing['frames'][0]['actor_loss_mask']
        partial_array = copy.deepcopy(complete)
        partial_array['frames'][0]['observations'] = []
        for data in ({}, dict(complete, frames=complete['frames'][:24]),
                     dict(complete, runtime_error='CUDA failure'), missing, partial_array):
            with self.subTest(keys=list(data)):
                self.assertEqual(evaluate(data)['status'], 'INCOMPLETE')

    def test_old_phase_d_evidence_cannot_be_admitted(self):
        path = os.path.join(os.path.dirname(__file__), '..', 'evaluation_assets',
                            'training_init_chain_gpu.json')
        with open(path) as fh:
            historical = json.load(fh)
        self.assertEqual(historical['envs_terminated_ids'], [0])
        self.assertNotEqual(evaluate(historical)['status'], 'PASS')

    def test_real_exit_codes_and_invalid_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'evidence.json')
            script = os.path.join(os.path.dirname(__file__), 'training_init_gate.py')
            failed = clean_evidence()
            failed['frames'][13]['done_env_ids'] = [0]
            for data, expected in ((clean_evidence(), 0), (failed, 1), ({}, 2)):
                with open(path, 'w') as fh:
                    json.dump(data, fh)
                result = subprocess.run([sys.executable, script, path], capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            with open(path, 'w') as fh:
                fh.write('broken JSON')
            self.assertEqual(subprocess.run([sys.executable, script, path], capture_output=True).returncode, 2)


if __name__ == '__main__':
    unittest.main()
