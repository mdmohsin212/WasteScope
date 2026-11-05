import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import builtins
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("model_setup",ROOT / "scripts/setup_models.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)

class ModelSetupTests(unittest.TestCase):
    def record(self, payload=b"selected-checkpoint"):
        return {
            "detector": {
                "filename": "detector.pt",
                "source_path": "runs/detect/D1/weights/best.pt",
                "sha256": hashlib.sha256(payload).hexdigest()
            }
        }

    def test_stage_only_selected_checkpoint_and_reuse_it(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source/runs/detect/D1/weights/best.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"selected-checkpoint")
            (root / "source/train.zip").write_bytes(b"not-needed")
            target = root / "models"

            setup.stage(self.record(), target, source_root=root / "source")
            source.unlink()
            
            setup.stage(self.record(), target, source_root=root / "source")

            self.assertEqual([p.name for p in target.iterdir()], ["detector.pt"])
            self.assertEqual((target / "detector.pt").read_bytes(), b"selected-checkpoint")

    def test_corrupt_copy_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source/runs/detect/D1/weights/best.pt"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"corrupt-source")

            target = root / "models"
            target.mkdir()
            (target / "detector.pt").write_bytes(b"previous-file")

            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                setup.stage(self.record(), target, source_root=root / "source")

            self.assertEqual((target / "detector.pt").read_bytes(), b"previous-file")
            self.assertEqual(len(list(target.iterdir())), 1)

    def test_rclone_receives_exact_source_and_config(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            received = []

            def copy(command, **kwargs):
                received.append(command)
                Path(command[-1]).write_bytes(b"selected-checkpoint")

            with (
                patch.object(setup.shutil, "which", return_value="rclone",),
                patch.object(setup.subprocess, "run", side_effect=copy)
            ):
                setup.stage(self.record(), root / "models", remote="gdrive:WasteScope", rclone_config=root / "private config/rclone.conf")

            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][3:5], ["copyto", "gdrive:WasteScope/runs/detect/D1/weights/best.pt"],)
            self.assertEqual(received[0][2],str(root / "private config/rclone.conf"))

    def test_check_works_without_importing_model_framework(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models").mkdir()
            (root / "models/manifest.json").write_text(json.dumps({"models": self.record()}))

            output = root / "weights"
            output.mkdir()
            (output / "detector.pt").write_bytes(b"selected-checkpoint")

            original = builtins.__import__
            def guarded(name, *args, **kwargs):
                if name.split(".")[0] in {"torch", "ultralytics"}:
                    raise RuntimeError("--check must not import inference frameworks")
                return original(name, *args, **kwargs)

            with (patch.object(setup, "ROOT", root), patch("builtins.__import__", side_effect=guarded)):
                setup.main(["--check", "--models-dir", str(output)])


if __name__ == "__main__":
    unittest.main()