import os
import sys
import unittest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cpu_affinity import (
    AFFINITY_ENV_VAR,
    ensure_affinity_env_default,
    parse_cpu_list_spec,
    resolve_affinity_mask,
)


class CpuAffinityTests(unittest.TestCase):
    def test_ensure_affinity_env_default_leaves_env_missing_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            value = ensure_affinity_env_default()
            self.assertEqual('', value)
            self.assertNotIn(AFFINITY_ENV_VAR, os.environ)

    def test_ensure_affinity_env_default_can_set_explicit_default(self):
        with patch.dict(os.environ, {}, clear=True):
            value = ensure_affinity_env_default('all')
            self.assertEqual('all', value)
            self.assertEqual('all', os.environ[AFFINITY_ENV_VAR])

    def test_ensure_affinity_env_default_preserves_existing_value(self):
        with patch.dict(os.environ, {AFFINITY_ENV_VAR: 'disabled'}, clear=True):
            value = ensure_affinity_env_default()
            self.assertEqual('disabled', value)
            self.assertEqual('disabled', os.environ[AFFINITY_ENV_VAR])

    def test_parse_cpu_list_spec_supports_ranges(self):
        mask = parse_cpu_list_spec('0,2-4,7')
        self.assertEqual(mask, (1 << 0) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 7))

    def test_resolve_p_cores_uses_highest_efficiency_class(self):
        mask, reason = resolve_affinity_mask(
            'p_cores',
            allowed_mask=0xFFFF,
            efficiency_class_masks={0: 0xF000, 8: 0x0FFF},
        )
        self.assertEqual(mask, 0x0FFF)
        self.assertIn('Windows efficiency class 8', reason)

    def test_resolve_p_cores_intersects_with_allowed_mask(self):
        mask, _ = resolve_affinity_mask(
            'p_cores',
            allowed_mask=0x03FF,
            efficiency_class_masks={0: 0xFC00, 8: 0x0FFF},
        )
        self.assertEqual(mask, 0x03FF)

    def test_resolve_p_cores_keeps_mask_when_system_is_not_hybrid(self):
        mask, reason = resolve_affinity_mask(
            'p_cores',
            allowed_mask=0x00FF,
            efficiency_class_masks={0: 0x00FF},
        )
        self.assertEqual(mask, 0x00FF)
        self.assertIn('single efficiency class', reason)


if __name__ == '__main__':
    unittest.main()
