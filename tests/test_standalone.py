"""Standalone build regression: missing default assets may not stop module import."""
import subprocess
import sys
import tempfile
import shutil
import unittest
from pathlib import Path


class TestStandaloneImport(unittest.TestCase):
    def test_explicit_missing_custom_file_still_errors(self):
        import ETNDNN as model
        with tempfile.TemporaryDirectory() as folder:
            custom = Path(folder) / "original_experiment.csv"
            with self.assertRaises(FileNotFoundError):
                model.load_figure_points(str(custom))

    def test_embedded_defaults_when_script_copied_alone(self):
        source = Path(__file__).resolve().parents[1] / "ETNDNN.py"
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "ETNDNN_stable_complete.py"
            shutil.copy2(source, target)
            proc = subprocess.run(
                [sys.executable, "-c",
                 "import ETNDNN_stable_complete as m; "
                 "assert len(m.FIGURE_POINTS_DF) == 16; "
                 "assert m.default_etn_params()['transition_uw_um2'] == 495.0"],
                cwd=folder, capture_output=True, text=True, timeout=40,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("NOT RAW EXPERIMENTAL", proc.stderr)
            self.assertFalse((Path(folder)/"data").exists())
            self.assertFalse((Path(folder)/"config").exists())


if __name__ == "__main__":
    unittest.main()
