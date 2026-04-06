import codecs
import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sl_ab as sl_ab
import toml_utils


class TomlUtilsTests(unittest.TestCase):
    def test_load_toml_file_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'config.toml'
            config_path.write_bytes(codecs.BOM_UTF8 + b"[control]\nversion = 4\n")

            cfg = toml_utils.load_toml_file(config_path)

            self.assertEqual(4, cfg['control']['version'])

    def test_write_toml_file_emits_utf8_without_bom(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'config.toml'

            toml_utils.write_toml_file(config_path, {'control': {'version': 4}})

            raw = config_path.read_bytes()
            self.assertFalse(raw.startswith(codecs.BOM_UTF8))
            self.assertIn(b'version = 4', raw)

    def test_config_module_accepts_utf8_bom_via_mortal_cfg(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'config.toml'
            config_path.write_bytes(
                codecs.BOM_UTF8
                + b"[control]\nversion = 4\n[supervised]\nstate_file = 'checkpoints/latest.pth'\n"
            )

            original_config_module = sys.modules.get('config')
            try:
                with patch.dict(os.environ, {'MORTAL_CFG': str(config_path)}, clear=False):
                    sys.modules.pop('config', None)
                    config_module = importlib.import_module('config')

                self.assertEqual(4, config_module.config['control']['version'])
                self.assertEqual(
                    str((config_path.parent / 'checkpoints' / 'latest.pth').resolve()),
                    config_module.config['supervised']['state_file'],
                )
            finally:
                sys.modules.pop('config', None)
                if original_config_module is not None:
                    sys.modules['config'] = original_config_module

    def test_mortal_config_module_imports_as_package(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'config.toml'
            config_path.write_bytes(
                codecs.BOM_UTF8
                + b"[control]\nversion = 4\n[supervised]\nstate_file = 'checkpoints/latest.pth'\n"
            )

            repo_root = Path(__file__).resolve().parent.parent
            original_mortal_module = sys.modules.get('mortal')
            original_mortal_config_module = sys.modules.get('mortal.config')
            original_mortal_toml_utils_module = sys.modules.get('mortal.toml_utils')
            original_sys_path = list(sys.path)
            mortal_dir = str(Path(__file__).resolve().parent)
            try:
                with patch.dict(os.environ, {'MORTAL_CFG': str(config_path)}, clear=False):
                    sys.modules.pop('mortal.config', None)
                    sys.modules.pop('mortal.toml_utils', None)
                    sys.modules.pop('mortal', None)
                    sys.path[:] = [path for path in sys.path if path != mortal_dir]
                    sys.path.insert(0, str(repo_root))
                    config_module = importlib.import_module('mortal.config')

                self.assertEqual(4, config_module.config['control']['version'])
                self.assertEqual(
                    str((config_path.parent / 'checkpoints' / 'latest.pth').resolve()),
                    config_module.config['supervised']['state_file'],
                )
            finally:
                sys.path[:] = original_sys_path
                sys.modules.pop('mortal.config', None)
                sys.modules.pop('mortal.toml_utils', None)
                sys.modules.pop('mortal', None)
                if original_mortal_module is not None:
                    sys.modules['mortal'] = original_mortal_module
                if original_mortal_config_module is not None:
                    sys.modules['mortal.config'] = original_mortal_config_module
                if original_mortal_toml_utils_module is not None:
                    sys.modules['mortal.toml_utils'] = original_mortal_toml_utils_module

    def test_build_base_config_accepts_utf8_bom(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            config_path = Path(tmp_dir) / 'config.toml'
            config_path.write_bytes(codecs.BOM_UTF8 + b"[control]\nversion = 4\n")
            original_base_cfg_path = sl_ab.BASE_CFG_PATH
            try:
                sl_ab.BASE_CFG_PATH = config_path
                cfg = sl_ab.build_base_config()
            finally:
                sl_ab.BASE_CFG_PATH = original_base_cfg_path

            self.assertEqual(4, cfg['control']['version'])


if __name__ == '__main__':
    unittest.main()
