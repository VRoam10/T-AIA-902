"""Register all benchmarks with the pipeline registry."""

from benchmarks.convergence import ConvergenceBenchmark
from core.registry import registry

registry.register_benchmark("convergence", ConvergenceBenchmark)
