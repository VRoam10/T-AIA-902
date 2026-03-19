"""Register all benchmarks with the pipeline registry."""

from core.registry import registry
from benchmarks.convergence import ConvergenceBenchmark

registry.register_benchmark("convergence", ConvergenceBenchmark)
