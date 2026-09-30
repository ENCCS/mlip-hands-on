#!/usr/bin/env python3
"""Check that the notebook adapter sends the visible cell without a shell."""

import os
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class Shell:
    def register_magic_function(self, function, kind, name):
        assert kind == "cell" and name == "alchemi"
        self.magic = function


class NotebookAdapterTests(unittest.TestCase):
    def test_visible_source_is_stdin_and_arguments_are_bounded(self):
        shell = Shell()
        with patch("IPython.get_ipython", return_value=shell):
            runpy.run_path(str(ROOT / "scripts/notebook_alchemi.py"))
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            sif = Path(directory) / "image.sif"
            model.touch()
            sif.touch()
            with patch.dict(os.environ, {"MLIP_MODEL": str(model),
                                             "MLIP_ALCHEMI_SIF": str(sif)}):
                with patch("subprocess.run") as run:
                    shell.magic("8 2000", "print('visible MD source')\n")
                command = run.call_args.args[0]
                self.assertEqual(command[:4], ["apptainer", "exec", "--cleanenv", "--nv"])
                self.assertEqual(command[-2:], ["python", "-"])
                self.assertIn("MLIP_REPLICAS=8", command)
                self.assertIn("MLIP_STEPS=2000", command)
                self.assertEqual(run.call_args.kwargs["input"], "print('visible MD source')\n")
                self.assertTrue(run.call_args.kwargs["check"])
            with self.assertRaises(ValueError):
                shell.magic("8; touch /tmp/unwanted 2000", "print(1)")


if __name__ == "__main__":
    unittest.main()
