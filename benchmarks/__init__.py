"""Register all benchmarks with the pipeline registry."""

from benchmarks.comparison import ComparisonBenchmark
from benchmarks.convergence import ConvergenceBenchmark
from core.registry import registry

registry.register_benchmark("convergence", ConvergenceBenchmark)
registry.register_benchmark("comparison", ComparisonBenchmark)
