import unittest

import torch

from mortal.supervised.metric_reporting import (
    ClusterMetricAccumulator,
    write_metric_scalars,
)


class FakeWriter:
    def __init__(self):
        self.scalars = []

    def add_scalar(self, name, value, step):
        self.scalars.append((name, value, step))


class MetricReportingTests(unittest.TestCase):
    def test_cluster_accumulator_merges_batches_and_sorts_games(self):
        accumulator = ClusterMetricAccumulator()
        accumulator.merge({
            'loss': (
                torch.tensor([2, 1, 2]),
                torch.tensor([0.5, 1.0, 1.5]),
            ),
        })
        accumulator.merge({
            'loss': (
                torch.tensor([1, 3]),
                torch.tensor([2.0, 4.0]),
            ),
        })

        self.assertEqual(
            {'loss': [[1, 3.0, 2], [2, 2.0, 2], [3, 4.0, 1]]},
            accumulator.records(),
        )

    def test_cluster_accumulator_rejects_mismatched_lengths(self):
        accumulator = ClusterMetricAccumulator()
        with self.assertRaisesRegex(ValueError, 'mismatched'):
            accumulator.merge({
                'loss': (torch.tensor([1]), torch.tensor([1.0, 2.0])),
            })

    def test_scalar_writer_emits_present_metrics_only(self):
        writer = FakeWriter()
        write_metric_scalars(
            writer,
            'monitor',
            {
                'loss': 1.5,
                'action_acc': 0.75,
                'discard_balanced_acc': 0.6,
            },
            100,
            decision_metric_names=('discard',),
        )

        self.assertEqual(
            [
                ('monitor/loss', 1.5, 100),
                ('monitor/action_acc', 0.75, 100),
                ('monitor/discard_balanced_acc', 0.6, 100),
            ],
            writer.scalars,
        )


if __name__ == '__main__':
    unittest.main()
