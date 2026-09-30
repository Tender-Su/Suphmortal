import unittest

from mortal.core.config_utils import (
    coerce_bool,
    deep_merge_dict,
    ensure_dict_section,
    get_dict_section,
)


class ConfigUtilsTests(unittest.TestCase):
    def test_read_only_section_does_not_mutate_missing_or_invalid_values(self):
        missing = {}
        invalid = {'value': 1}

        self.assertEqual({}, get_dict_section(missing, 'value'))
        self.assertEqual({}, missing)
        self.assertEqual({}, get_dict_section(invalid, 'value'))
        self.assertEqual({'value': 1}, invalid)

    def test_ensure_section_reuses_or_replaces_values(self):
        payload = {'existing': {'value': 1}, 'invalid': 2}

        self.assertIs(payload['existing'], ensure_dict_section(payload, 'existing'))
        replacement = ensure_dict_section(payload, 'invalid')

        self.assertIs(payload['invalid'], replacement)
        self.assertEqual({}, replacement)

    def test_deep_merge_copies_values_and_replaces_invalid_sections(self):
        source = {'nested': {'items': [1, 2]}, 'value': 3}
        destination = {'nested': 1, 'keep': True}

        deep_merge_dict(destination, source)
        source['nested']['items'].append(4)

        self.assertEqual(
            {'nested': {'items': [1, 2]}, 'value': 3, 'keep': True},
            destination,
        )

    def test_coerce_bool_accepts_common_values_and_preserves_default(self):
        self.assertTrue(coerce_bool('YES'))
        self.assertFalse(coerce_bool('off', default=True))
        self.assertTrue(coerce_bool('unknown', default=True))


if __name__ == '__main__':
    unittest.main()
