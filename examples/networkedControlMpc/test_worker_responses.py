"""Test worker-response diagnostics with ordinary subprocesses, without NumPy.

Only the helper's AST is loaded from the driver. Importing the full driver
would unnecessarily require the simulation's numerical and GPU packages.
"""
import ast
import json
from pathlib import Path
import signal
import subprocess
import sys
import unittest


def load_response_helper():
    source_path = Path(__file__).with_name("run_mpc_persistent.py")
    syntax = ast.parse(source_path.read_text(), filename=str(source_path))
    helper = next(node for node in syntax.body
                  if isinstance(node, ast.FunctionDef) and node.name == "read_worker_response")
    namespace = {"json": json, "signal": signal, "subprocess": subprocess}
    code = compile(ast.Module(body=[helper], type_ignores=[]), str(source_path), "exec")
    exec(code, namespace)
    return namespace[helper.name]


class WorkerResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.read_response = staticmethod(load_response_helper())

    def start_worker(self, source):
        worker = subprocess.Popen([sys.executable, "-c", source], stdout=subprocess.PIPE)
        self.addCleanup(self.close_worker, worker)
        return worker

    @staticmethod
    def close_worker(worker):
        # These are test-created children only; no simulation is signalled.
        if worker.poll() is None:
            worker.terminate()
        worker.wait(timeout=5)
        worker.stdout.close()

    def test_valid_json_response(self):
        worker = self.start_worker('import json,time; print(json.dumps({"ready": True, "iterations": 25}), flush=True); time.sleep(10)')
        self.assertEqual(self.read_response(worker, "initialisation"), {"ready": True, "iterations": 25})
        self.assertIsNone(worker.poll())  # A normal reply must not wait for process exit.

    def test_eof_reports_actual_exit_code(self):
        # Explicitly close stdout first, reproducing the EOF/poll race.
        worker = self.start_worker('import os,time; os.close(1); time.sleep(0.05); raise SystemExit(7)')
        with self.assertRaisesRegex(RuntimeError, "return code 7.*initialisation response"):
            self.read_response(worker, "initialisation")
        self.assertEqual(worker.returncode, 7)

    @unittest.skipUnless(hasattr(signal, "SIGTERM"), "Requires POSIX termination signals")
    def test_eof_identifies_signal_termination(self):
        worker = self.start_worker('import os,signal; os.kill(os.getpid(), signal.SIGTERM)')
        with self.assertRaisesRegex(RuntimeError, r"return code -15 \(SIGTERM\).*solve response"):
            self.read_response(worker, "solve")
        self.assertEqual(worker.returncode, -signal.SIGTERM)

    def test_eof_distinguishes_a_child_that_is_still_alive(self):
        worker = self.start_worker('import os,time; os.close(1); time.sleep(10)')
        with self.assertRaisesRegex(RuntimeError, "closed stdout.*still running"):
            self.read_response(worker, "solve", exit_wait_seconds=0.02)
        self.assertIsNone(worker.poll())


if __name__ == "__main__":
    unittest.main(verbosity=2)
