"""Profile four real updates after ordered preparation has warmed up.

This is a CPU attribution pass, not a throughput ranking. Production native,
objective, microbatches and gradients use the same isolated benchmark entry.
"""
import cProfile
from pathlib import Path
import pstats
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mortal.core.artifacts import atomic_write_json
from scripts import benchmark_sl_ordered_preparation as benchmark


class ProfileProbe(benchmark.BenchmarkProbe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.profile = cProfile.Profile()
        self.profile_start = None
        self.record['format'] = 'sl_ordered_preparation_cpu_profile_v1'

    def after_update(self, optimizer_steps, *args):
        if self.profile_start is not None and optimizer_steps == self.profile_start + 4:
            self.profile.disable()
            self.profile.dump_stats(str(self.output / 'training_calls.pstats'))
            rows = []
            for (filename, line, name), (primitive, calls, own, cumulative, _) in pstats.Stats(self.profile).stats.items():
                rows.append({'file': filename, 'line': line, 'name': name, 'primitive_calls': primitive,
                             'calls': calls, 'self_s': own, 'cumulative_s': cumulative})
            atomic_write_json(self.output / 'training_cpu_profile.json', {
                'first_update': self.profile_start + 1, 'last_update': optimizer_steps,
                'self_time': sorted(rows, key=lambda row: -row['self_s'])[:50],
                'cumulative_time': sorted(rows, key=lambda row: -row['cumulative_s'])[:35],
                'limitation': 'CPU call wall time includes device waits; instrumented timings do not rank configurations.'})
        stop = super().after_update(optimizer_steps, *args)
        if self.profile_start is None and optimizer_steps >= self.warmup + 1:
            self.profile_start = optimizer_steps
            self.profile.enable()
        return stop


if __name__ == '__main__':
    benchmark.BenchmarkProbe = ProfileProbe
    benchmark.main()
