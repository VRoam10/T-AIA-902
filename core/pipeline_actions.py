"""UI-driven action layer for the RL pipeline.

This module is the single place that knows how to construct agents,
environments, benchmark configs, and multi-agent slots from declarative
requests. It carries no interactive prompts: every decision arrives as a
dataclass or plain dict produced by the TUI, so the same functions are used by
the backend bridge and by tests.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from pathlib import Path

import algorithms  # noqa: F401 — triggers registry auto-registration
import benchmarks  # noqa: F401 — triggers registry auto-registration
import environments  # noqa: F401 — triggers registry auto-registration
from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
from core.multi_runner import MultiAgentRunner
from core.registry import registry
from core.runner import PipelineRunner
from environments.beamng_multi import BeamNGMultiEnv, build_slots, slot_n_states

BEAMNG_MAPS: tuple[str, ...] = ("gridmap_v2", "italy", "west_coast_usa")

BEAMNG_VEHICLES: dict[str, str] = {
    "taxi": "Burnside (Taxi)",
    "gavril_t_series": "Gavril T-Series",
    "ibishu_pigeon": "Ibishu Pigeon",
    "gavril_d_series": "Gavril D-Series",
}

MULTI_ALGOS: tuple[str, ...] = ("dqn", "ddpg", "td3")

# Multi-agent checkpoints + plots live in their own subfolder.
MULTI_OUTPUT_DIR = os.path.join("outputs", "multi-agents")

_TRAJECTORY_DIR = os.path.join("outputs", "trajectories")


# --------------------------------------------------------------------------- #
# Formatting / parsing helpers
# --------------------------------------------------------------------------- #
def format_trajectory_summary(mt) -> str:
    """One-line summary of a MapTrajectories: path count + per-path sources."""
    n = len(mt.paths)
    sources = ", ".join(p.source for p in mt.paths)
    return f"{mt.map_name}: {n} path(s) [{sources}]"


def parse_values(raw: str) -> list:
    """Parse a comma-separated string into ints/floats/strings."""
    values = []
    for token in raw.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            values.append(float(token) if ("." in token or "e" in token.lower()) else int(token))
        except ValueError:
            values.append(token)
    return values


def format_aggregate(multi_results: dict) -> str:
    """Format the headline aggregated metrics of a multi-seed run."""
    agg = multi_results["aggregate"]
    lines = [f"=== multi-seed ({multi_results['n_seeds']} seeds) ==="]
    for key in ("eval_mean_reward", "eval_success_rate", "convergence_episode", "training_time_s"):
        if key in agg:
            stat = agg[key]
            lines.append(f"  {key}: {stat['mean']} +/- {stat['std']} (ci95 {stat['ci95']})")
    return "\n".join(lines)


def trajectory_cache_path(
    map_name: str, output_dir: str | os.PathLike[str] = _TRAJECTORY_DIR
) -> str:
    """Resolve the per-map trajectory cache file path."""
    return str(Path(output_dir) / f"{map_name}.json")


# --------------------------------------------------------------------------- #
# Request dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class BeamNGOptions:
    map_name: str = "gridmap_v2"
    vehicle_id: str = "taxi"
    trajectory_hints: int = 0
    body_orientation: bool = False
    wheel_terrain: bool = False
    random_path: bool = False
    dense_episodes: int = 0


@dataclass
class TrainRequest:
    algo_name: str
    env_name: str
    n_episodes: int
    save_path: str
    agent_params: dict[str, object]
    beamng: BeamNGOptions | None = None
    reset_existing: bool = False


@dataclass
class EvaluateRequest:
    algo_name: str
    env_name: str
    model_path: str
    n_episodes: int
    beamng: BeamNGOptions | None = None


@dataclass
class BenchmarkRequest:
    benchmark_name: str
    seeds: list[int]
    eval_episodes: int
    success_threshold: float
    max_episodes: int
    reward_threshold: float = 7.0
    algo_name: str | None = None
    env_name: str | None = None
    algos: list[str] | None = None
    param_grid: dict[str, list] | None = None


@dataclass
class HumanPlayRequest:
    map_name: str
    vehicle_id: str
    sensor: str = "None"
    random_path: bool = False


@dataclass
class TrajectoryRequest:
    map_name: str
    overwrite: bool = False


@dataclass
class MultiTrainRequest:
    map_name: str
    random_path: bool
    specs: list[dict]
    n_episodes: int
    time_limit_minutes: float = 0.0
    reset_existing: bool = False


# --------------------------------------------------------------------------- #
# Catalog
# --------------------------------------------------------------------------- #
def catalog() -> dict[str, object]:
    """Return a JSON-serializable description of the registered pipeline.

    Only primitive data is returned: no classes or factories. Used by the TUI
    to build forms without importing Python.
    """
    algos = []
    compatible = {}
    for name in registry.list_algorithms():
        info = registry.get_algorithm(name)
        envs = registry.compatible_environments(name)
        compatible[name] = envs
        algos.append(
            {
                "name": name,
                "default_config": dict(info["default_config"]),
                "compatible_envs": info.get("compatible_envs"),
            }
        )

    envs = [
        {"name": name, "metadata": dict(registry.get_environment(name)["metadata"])}
        for name in registry.list_environments()
    ]

    return {
        "algorithms": algos,
        "environments": envs,
        "compatible_envs": compatible,
        "benchmarks": registry.list_benchmarks(),
        "beamng_maps": list(BEAMNG_MAPS),
        "beamng_vehicles": [{"id": k, "label": v} for k, v in BEAMNG_VEHICLES.items()],
        "multi_algos": list(MULTI_ALGOS),
    }


# --------------------------------------------------------------------------- #
# Agent construction
# --------------------------------------------------------------------------- #
def build_agent(
    algo_name: str,
    env_name: str,
    agent_params: dict[str, object] | None = None,
    beamng: BeamNGOptions | None = None,
):
    """Instantiate an agent sized to the chosen env + BeamNG observation flags.

    Mirrors the old ``_build_agent`` sizing: env metadata supplies the base
    observation length, BeamNG flags add extra dims, and ``agent_params`` are
    merged over the algorithm defaults.
    """
    algo_info = registry.get_algorithm(algo_name)
    env_metadata = registry.get_environment(env_name)["metadata"]
    cls = algo_info["class"]
    params = dict(algo_info["default_config"])

    if beamng is not None:
        extra = (
            beamng.trajectory_hints * 2
            + (2 if beamng.body_orientation else 0)
            + (2 if beamng.wheel_terrain else 0)
        )
    else:
        extra = 0

    params["n_states"] = env_metadata["n_states"] + extra
    state_type = env_metadata.get("state_type", "continuous")
    params["state_type"] = state_type
    if state_type == "discrete":
        # A discrete env fixes the action count, so it overrides any
        # continuous-control default (e.g. TD3's n_actions=2).
        params["n_actions"] = env_metadata.get("n_actions")
    else:
        params.setdefault("n_actions", env_metadata.get("n_actions"))

    if agent_params:
        params.update(agent_params)

    # Only agents whose constructor accepts ``state_type`` (the continuous-control
    # agents, which switch to discrete one-hot / argmax mode for Taxi) receive it.
    if "state_type" not in inspect.signature(cls).parameters:
        params.pop("state_type", None)
    return cls(**params)


def _beamng_kwargs(beamng: BeamNGOptions | None, *, with_random_path: bool) -> dict:
    if beamng is None:
        return {}
    kwargs = {
        "map_name": beamng.map_name,
        "vehicle_id": beamng.vehicle_id,
        "trajectory_hints": beamng.trajectory_hints,
        "body_orientation": beamng.body_orientation,
        "wheel_terrain": beamng.wheel_terrain,
    }
    if with_random_path:
        # Training-only options: evaluation always runs sparse checkpoints so
        # the dense warm-up curriculum never leaks into eval metrics.
        kwargs["random_path"] = beamng.random_path
        kwargs["dense_episodes"] = beamng.dense_episodes
    return kwargs


# --------------------------------------------------------------------------- #
# Train / Evaluate
# --------------------------------------------------------------------------- #
def run_train(request: TrainRequest) -> dict[str, object]:
    agent = build_agent(request.algo_name, request.env_name, request.agent_params, request.beamng)

    env_info = registry.get_environment(request.env_name)
    beamng_kwargs = _beamng_kwargs(request.beamng, with_random_path=True)

    # Continuous-action algorithms get their own reward mode.
    if request.beamng is not None or request.env_name.startswith("beamng"):
        reward_mode = request.algo_name if request.algo_name in ("ddpg", "td3") else "default"
        env = env_info["factory"](reward_mode=reward_mode, **beamng_kwargs)
    else:
        env = env_info["factory"]()

    save_path = request.save_path
    plot_path = f"outputs/{request.algo_name}_{request.env_name}_training.png"
    start_episode = 0

    if os.path.exists(save_path):
        if request.reset_existing:
            os.remove(save_path)
        else:
            agent.load(save_path)
            start_episode = getattr(agent, "episode", 0)

    os.makedirs("outputs", exist_ok=True)
    runner = PipelineRunner()
    try:
        history = runner.train(
            agent,
            env,
            n_episodes=request.n_episodes,
            save_path=save_path,
            plot_path=plot_path,
            start_episode=start_episode,
        )
    finally:
        env.close()

    rewards = history.get("rewards", []) if isinstance(history, dict) else []
    return {
        "status": "ok",
        "save_path": save_path,
        "start_episode": start_episode,
        "episodes": len(rewards),
        "final_reward": rewards[-1] if rewards else None,
    }


def run_evaluate(request: EvaluateRequest) -> dict[str, object]:
    if not os.path.exists(request.model_path):
        raise FileNotFoundError(request.model_path)

    agent = build_agent(request.algo_name, request.env_name, None, request.beamng)
    agent.load(request.model_path)

    env_info = registry.get_environment(request.env_name)
    beamng_kwargs = _beamng_kwargs(request.beamng, with_random_path=False)
    if request.beamng is not None or request.env_name.startswith("beamng"):
        env = env_info["factory"](**beamng_kwargs)
    else:
        env = env_info["factory"]()

    runner = PipelineRunner()
    try:
        metrics = runner.evaluate(agent, env, n_episodes=request.n_episodes)
    finally:
        env.close()

    result: dict[str, object] = {"status": "ok", "model_path": request.model_path}
    if isinstance(metrics, dict):
        result["metrics"] = metrics
    return result


# --------------------------------------------------------------------------- #
# Benchmark
# --------------------------------------------------------------------------- #
def _validate_benchmark(request: BenchmarkRequest) -> None:
    name = request.benchmark_name
    missing: list[str] = []
    if name == "comparison":
        if not request.algos:
            missing.append("algos")
        if not request.env_name:
            missing.append("env_name")
    elif name == "gridsearch":
        if not request.algo_name:
            missing.append("algo_name")
        if not request.env_name:
            missing.append("env_name")
        if not request.param_grid:
            missing.append("param_grid")
    else:
        if not request.algo_name:
            missing.append("algo_name")
        if not request.env_name:
            missing.append("env_name")
    if missing:
        raise ValueError(f"Missing required fields for '{name}': {', '.join(missing)}")


def run_benchmark(request: BenchmarkRequest) -> dict[str, object]:
    _validate_benchmark(request)
    bench = registry.get_benchmark(request.benchmark_name)["class"]()
    common = {
        "seeds": request.seeds,
        "eval_episodes": request.eval_episodes,
        "success_threshold": request.success_threshold,
        "max_episodes": request.max_episodes,
    }

    if request.benchmark_name == "comparison":
        return _run_comparison(bench, request, common)
    if request.benchmark_name == "gridsearch":
        return _run_gridsearch(bench, request, common)
    return _run_single(bench, request, common)


def _run_single(bench, request: BenchmarkRequest, common: dict) -> dict[str, object]:
    algo_info = registry.get_algorithm(request.algo_name)
    env_info = registry.get_environment(request.env_name)
    config = {
        "agent_params": algo_info["default_config"],
        "env_metadata": env_info["metadata"],
        "threshold": request.reward_threshold,
        **common,
    }
    seeds = common["seeds"]
    if len(seeds) > 1:
        results = bench.run_multi(algo_info["class"], env_info["factory"], config)
        bench.export_multi(results, request.algo_name, request.env_name)
        return {"status": "ok", "summary": format_aggregate(results)}
    config["seed"] = seeds[0]
    results = bench.run(algo_info["class"], env_info["factory"], config)
    bench.export(results, request.algo_name, request.env_name)
    return {"status": "ok", "report": bench.report(results)}


def _run_comparison(bench, request: BenchmarkRequest, common: dict) -> dict[str, object]:
    algos = request.algos
    env_info = registry.get_environment(request.env_name)
    config = {
        "env_metadata": env_info["metadata"],
        "threshold": request.reward_threshold,
        "window": 100,
        "variants": [{"name": algo, "algo": algo} for algo in algos],
        **common,
    }
    results = bench.run(None, env_info["factory"], config)
    lines = []
    for label, data in results["variants"].items():
        stat = data["aggregate"].get("eval_mean_reward", {})
        lines.append(f"  {label}: eval reward {stat.get('mean', 0)} +/- {stat.get('std', 0)}")
    bench.export(results, "+".join(algos), request.env_name)
    return {"status": "ok", "summary": "\n".join(lines)}


def _run_gridsearch(bench, request: BenchmarkRequest, common: dict) -> dict[str, object]:
    algo_info = registry.get_algorithm(request.algo_name)
    env_info = registry.get_environment(request.env_name)
    config = {
        "agent_params": algo_info["default_config"],
        "env_metadata": env_info["metadata"],
        "param_grid": request.param_grid,
        **common,
    }
    results = bench.run(algo_info["class"], env_info["factory"], config)
    bench.export(results, request.algo_name, request.env_name)
    return {"status": "ok", "report": bench.report(results)}


# --------------------------------------------------------------------------- #
# Human play
# --------------------------------------------------------------------------- #
def run_human_play(request: HumanPlayRequest) -> None:
    env = None
    try:
        if request.sensor == "Camera":
            env = registry.get_environment("beamng_camera")["factory"](
                map_name=request.map_name,
                vehicle_id=request.vehicle_id,
                random_path=request.random_path,
            )
        else:
            env = registry.get_environment("beamng")["factory"](
                map_name=request.map_name,
                vehicle_id=request.vehicle_id,
                random_path=request.random_path,
            )

        if request.sensor == "LiDAR":
            env.human_play_lidar()
        elif request.sensor == "Camera":
            env.human_play_camera()
        else:
            env.human_play()
    finally:
        if env is not None:
            env.close(kill_sim=True)


# --------------------------------------------------------------------------- #
# Trajectory
# --------------------------------------------------------------------------- #
def run_trajectory(
    request: TrajectoryRequest, output_dir: str | os.PathLike[str] = _TRAJECTORY_DIR
) -> dict[str, object]:
    cache_path = trajectory_cache_path(request.map_name, output_dir)

    if os.path.exists(cache_path):
        if not request.overwrite:
            return {"status": "skipped", "path": cache_path}
        os.remove(cache_path)

    import core.trajectory as trajectory_mod
    from core.trajectory import load_or_generate
    from environments.beamng import BeamNGDrivingEnv

    original_cache_dir = trajectory_mod.CACHE_DIR
    env = None
    try:
        trajectory_mod.CACHE_DIR = Path(output_dir)
        env = BeamNGDrivingEnv(
            beamng_home=BEAMNG_HOME,
            beamng_user=BEAMNG_USER,
            headless=HEADLESS,
            map_name=request.map_name,
        )
        env.reset()
        mt = load_or_generate(request.map_name, bng=None)
        summary = format_trajectory_summary(mt)
        return {"status": "ok", "path": cache_path, "summary": summary}
    finally:
        if env is not None:
            env.close()
        trajectory_mod.CACHE_DIR = original_cache_dir


# --------------------------------------------------------------------------- #
# Multi-agent
# --------------------------------------------------------------------------- #
def build_multi_session(specs: list[dict], map_name: str, random_path: bool = False):
    """Create the BeamNGMultiEnv and an agent per spec.

    Each spec carries its own ``env`` name; the agent is sized to that env's
    observation length (``slot_n_states``) and its action dimensionality from
    the algorithm's registered defaults. Returns (env, slots).
    """
    env = BeamNGMultiEnv(
        slots=[],
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        map_name=map_name,
        random_path=random_path,
    )

    enriched = []
    for spec in specs:
        algo_info = registry.get_algorithm(spec["algo"])
        cls = algo_info["class"]
        cfg = dict(algo_info["default_config"])
        trajectory_hints = spec.get("trajectory_hints", 0)
        body_orientation = spec.get("body_orientation", False)
        wheel_terrain = spec.get("wheel_terrain", False)
        cfg["n_states"] = slot_n_states(
            spec.get("env", "beamng"), trajectory_hints, body_orientation, wheel_terrain
        )
        # Discrete (DQN) uses the 7-action table; continuous algos keep their
        # configured action dimensionality (n_actions from defaults, else 3).
        if spec["algo"] == "dqn":
            cfg["n_actions"] = BeamNGMultiEnv.N_ACTIONS_DISCRETE
        else:
            cfg.setdefault("n_actions", 3)
        cfg.pop("state_type", None)
        agent = cls(**cfg)
        enriched.append({**spec, "agent": agent})

    slots = build_slots(enriched)
    env.slots = slots
    return env, slots


def run_multi_train(request: MultiTrainRequest) -> dict[str, object]:
    env, slots = build_multi_session(request.specs, request.map_name, request.random_path)

    for slot in slots:
        if os.path.exists(slot.save_path):
            if request.reset_existing:
                os.remove(slot.save_path)
            else:
                slot.agent.load(slot.save_path)
                slot.episode = getattr(slot.agent, "episode", 0)

    os.makedirs(MULTI_OUTPUT_DIR, exist_ok=True)
    time_limit = request.time_limit_minutes * 60.0 if request.time_limit_minutes > 0 else None
    runner = MultiAgentRunner()
    try:
        result = runner.train(env, n_episodes=request.n_episodes, time_limit=time_limit)
    finally:
        env.close()

    out: dict[str, object] = {"status": "ok", "n_agents": len(slots)}
    if isinstance(result, dict):
        out["result"] = result
    return out
