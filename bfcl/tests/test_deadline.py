import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import main


class DeadlineTests(unittest.TestCase):
    def test_deadline_stops_writer_and_preserves_completed_rows(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'result/model/BFCL_v4_simple_python_result.json'
            path.parent.mkdir(parents=True)
            def generate(*args):
                path.write_text(json.dumps({'id': 'simple_python_0', 'result': 'ok'}) + '\n{"id":')
                time.sleep(5)
                path.write_text('late write')
            with patch.object(main.shim, 'run_cli', generate):
                started = time.monotonic()
                self.assertFalse(main.run_generation_until([], root, 'model', 'model', 0.2))
                self.assertLess(time.monotonic() - started, 2)
            main.mark_timeouts(root, 'model', {'simple_python': ['simple_python_0', 'simple_python_1'],
                                             'multi_turn_base': ['multi_turn_base_0']})
            rows, failed = main.scan_results(root, 'model')
            self.assertEqual(rows['simple_python'][0]['result'], 'ok')
            self.assertEqual(rows['simple_python'][1]['status'], 'timeout')
            self.assertEqual(rows['multi_turn_base'][0]['status'], 'timeout')
            self.assertEqual(sum(map(len, failed.values())), 2)
            main.mark_timeouts(root, 'model', {'simple_python': ['simple_python_0', 'simple_python_1']})
            self.assertEqual(len(main.scan_results(root, 'model')[0]['simple_python']), 2)

    def test_fast_completion_and_failure_are_not_timeouts(self):
        with patch.object(main.shim, 'run_cli', lambda *args: None):
            self.assertTrue(main.run_generation_until([], '/tmp', 'model', 'model', 2))
        def fail(*args):
            raise RuntimeError('test failure')
        with patch.object(main.shim, 'run_cli', fail):
            with self.assertRaises(RuntimeError):
                main.run_generation_until([], '/tmp', 'model', 'model', 2)


if __name__ == '__main__':
    unittest.main()
