from custom_eval.benchmarks import code_tasks, if_tasks, math_tasks
from custom_eval.benchmarks._common import Benchmark, BuildContext

BENCHMARKS: dict[str, Benchmark] = {
    b.name: b
    for b in [*math_tasks.BENCHMARKS, *code_tasks.BENCHMARKS, *if_tasks.BENCHMARKS]
}
"""All benchmarks of the suite, by config name."""

__all__ = ["BENCHMARKS", "Benchmark", "BuildContext"]
