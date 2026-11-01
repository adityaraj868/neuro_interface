import unittest

from muse.event_mapping import labels_for_run
from muse.data.physionet import labels_from_event_descriptions, parse_subject_run


class EventMappingTests(unittest.TestCase):
    def test_unilateral_imagery_mapping(self):
        self.assertEqual(dict(labels_for_run(4)), {"T1": "LEFT", "T2": "RIGHT"})
        self.assertEqual(labels_from_event_descriptions(12, ["T1", "T2"]).tolist(), ["LEFT", "RIGHT"])

    def test_bilateral_mapping_is_not_directional_mapping(self):
        self.assertEqual(dict(labels_for_run(6)), {"T1": "BOTH_HANDS", "T2": "BOTH_FEET"})

    def test_executed_and_baseline_runs_are_rejected(self):
        with self.assertRaises(ValueError):
            labels_for_run(3)
        with self.assertRaises(ValueError):
            labels_for_run(1)

    def test_parse_filename(self):
        self.assertEqual(parse_subject_run("S007/S007R12.edf"), (7, 12))
        with self.assertRaises(ValueError):
            parse_subject_run("recording.edf")

