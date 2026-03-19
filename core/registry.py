from collections.abc import Callable


class Registry:
    """Central registry for algorithms, environments, and benchmarks."""

    def __init__(self):
        self._algorithms: dict = {}
        self._environments: dict = {}
        self._benchmarks: dict = {}

    # -- Algorithms ----------------------------------------------------------

    def register_algorithm(
        self,
        name: str,
        cls: type,
        default_config: dict = None,
        compatible_envs: list[str] | None = None,
    ):
        self._algorithms[name] = {
            "class": cls,
            "default_config": default_config or {},
            "compatible_envs": compatible_envs,
        }

    def get_algorithm(self, name: str) -> dict:
        return self._algorithms[name]

    def list_algorithms(self) -> list[str]:
        return list(self._algorithms.keys())

    # -- Environments --------------------------------------------------------

    def register_environment(
        self,
        name: str,
        factory: Callable,
        metadata: dict = None,
    ):
        self._environments[name] = {
            "factory": factory,
            "metadata": metadata or {},
        }

    def get_environment(self, name: str) -> dict:
        return self._environments[name]

    def list_environments(self) -> list[str]:
        return list(self._environments.keys())

    # -- Benchmarks ----------------------------------------------------------

    def register_benchmark(self, name: str, cls: type):
        self._benchmarks[name] = {"class": cls}

    def get_benchmark(self, name: str) -> dict:
        return self._benchmarks[name]

    def list_benchmarks(self) -> list[str]:
        return list(self._benchmarks.keys())

    # -- Filtering -----------------------------------------------------------

    def compatible_environments(self, algo_name: str) -> list[str]:
        """Return environment names compatible with the given algorithm."""
        algo = self._algorithms[algo_name]
        compat = algo.get("compatible_envs")
        if compat is None:
            return self.list_environments()
        return [e for e in compat if e in self._environments]


registry = Registry()
