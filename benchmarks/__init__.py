"""Register all benchmarks with the pipeline registry."""

from benchmarks.comparison import ComparisonBenchmark
from benchmarks.convergence import ConvergenceBenchmark
from benchmarks.gridsearch import GridSearchBenchmark
from core.registry import registry

registry.register_benchmark("convergence", ConvergenceBenchmark)
registry.register_benchmark("comparison", ComparisonBenchmark)
registry.register_benchmark("gridsearch", GridSearchBenchmark)
