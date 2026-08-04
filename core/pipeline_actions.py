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
from environments import beamng_spec
from environments.beamng_multi import BeamNGMultiEnv, build_slots, slot_n_states

BEAMNG_MAPS: tuple[str, ...] = beamng_spec.AVAILABLE_MAPS

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
    sensor: str = beamng_spec.DEFAULT_SENSOR
    trajectory_hints: int = 0
    body_orientation: bool = False
    road_info: bool = False
    random_path: bool = False
    dense_episodes: int = 0
    # One of the game's own race tracks (a core.quickrace key), or "" for the
    # generated road-network paths.
    track: str = ""


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
    beamng: BeamNGOptions | None = None


@dataclass
class HumanPlayRequest:
    map_name: str
    sensor: str = beamng_spec.DEFAULT_SENSOR
    random_path: bool = False
    track: str = ""


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
    track: str = ""


@dataclass
class RacerSpec:
    """One entrant in a race.

    A human entrant sets ``human=True`` and needs nothing else: the player drives,
    so there is no algorithm, checkpoint, sensor or action head to configure.
    """

    algo: str = ""
    sensor: str = beamng_spec.DEFAULT_SENSOR
    model_path: str = ""
    color: str = "White"
    trajectory_hints: int = 0
    body_orientation: bool = False
    road_info: bool = False
    human: bool = False


@dataclass
class CourseRequest:
    map_name: str
    racers: list[RacerSpec]
    laps: int = 1
    races: int = 1
    learning: bool = False
    path_idx: int = 0
    # One of the game's own race tracks (a core.quickrace key) to race on, or ""
    # to race generated path `path_idx`.
    track: str = ""


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
        "beamng_sensors": list(beamng_spec.SENSORS),
        "multi_algos": list(MULTI_ALGOS),
        "beamng_tracks": beamng_tracks(),
    }


def beamng_tracks() -> dict[str, list[dict[str, object]]]:
    """The game's race tracks per map, for the TUI's track picker.

    Read from the level archives, so this costs no simulator launch and works with
    the game closed. Returns ``{map_name: [{key, kind, checkpoints, length_m}]}``,
    longest track first. A map with no readable tracks maps to an empty list —
    a missing or moved BeamNG install must not break the catalog, since every other
    workflow still works without it.
    """
    from core import quickrace

    out: dict[str, list[dict[str, object]]] = {}
    for map_name in BEAMNG_MAPS:
        try:
            races = quickrace.load_all(map_name, BEAMNG_HOME)
        except Exception:  # noqa: BLE001 — an unreadable level costs its tracks only
            races = []
        out[map_name] = [
            {
                "key": r.key,
                "kind": r.kind,
                "checkpoints": len(r.checkpoints),
                "length_m": round(r.length_m()),
            }
            for r in races
        ]
    return out


# --------------------------------------------------------------------------- #
# Agent construction
# --------------------------------------------------------------------------- #
def build_agent(
    algo_name: str,
    env_name: str,
    agent_params: dict[str, object] | None = None,
    beamng: BeamNGOptions | None = None,
):
    """Instantiate an agent sized to the chosen sensor, options and algorithm.

    Sizes come from :mod:`environments.beamng_spec` rather than registry metadata:
    the observation length depends on the sensor plus the optional observation
    flags, and the action count on the output axis the algorithm implies. Explicit
    ``agent_params`` are merged over the algorithm defaults last.
    """
    algo_info = registry.get_algorithm(algo_name)
    env_metadata = registry.get_environment(env_name)["metadata"]
    cls = algo_info["class"]
    params = dict(algo_info["default_config"])

    options = beamng or BeamNGOptions()
    params["n_states"] = beamng_spec.obs_size(
        options.sensor,
        options.trajectory_hints,
        options.body_orientation,
        options.road_info,
    )
    params["n_actions"] = beamng_spec.action_size(beamng_spec.output_for_algo(algo_name))
    params["state_type"] = env_metadata.get("state_type", "continuous")

    if agent_params:
        params.update(agent_params)

    # Only the continuous-control agents accept ``state_type``; DQN does not.
    if "state_type" not in inspect.signature(cls).parameters:
        params.pop("state_type", None)
    return cls(**params)


def _beamng_kwargs(
    beamng: BeamNGOptions | None, algo_name: str | None = None, *, with_random_path: bool
) -> dict:
    """Env-factory kwargs for a BeamNG request.

    ``output`` is derived from the algorithm, so the env's action head can never
    disagree with the agent's. Pass ``algo_name=None`` to keep the factory default.
    """
    if beamng is None:
        return {}
    kwargs = {
        "map_name": beamng.map_name,
        "sensor": beamng.sensor,
        "trajectory_hints": beamng.trajectory_hints,
        "body_orientation": beamng.body_orientation,
        "road_info": beamng.road_info,
    }
    if beamng.track:
        # A chosen game track replaces the generated paths entirely, so
        # random_path (which deals a random generated path per episode) has
        # nothing to choose from and is left off.
        kwargs["track"] = beamng.track
    if algo_name is not None:
        kwargs["output"] = beamng_spec.output_for_algo(algo_name)
    if with_random_path:
        # Training-only options: evaluation always runs sparse checkpoints so
        # the dense warm-up curriculum never leaks into eval metrics.
        kwargs["random_path"] = beamng.random_path and not beamng.track
        kwargs["dense_episodes"] = beamng.dense_episodes
    return kwargs


# --------------------------------------------------------------------------- #
# Train / Evaluate
# --------------------------------------------------------------------------- #
def run_train(request: TrainRequest) -> dict[str, object]:
    agent = build_agent(request.algo_name, request.env_name, request.agent_params, request.beamng)

    env_info = registry.get_environment(request.env_name)
    env = env_info["factory"](
        **_beamng_kwargs(
            request.beamng or BeamNGOptions(), request.algo_name, with_random_path=True
        )
    )

    save_path = request.save_path
    # The plot belongs to the checkpoint it describes, so its name is derived from
    # ``save_path`` — the only name the caller chose. An independent algo/env stem
    # cannot match it (the model name carries the sensor and option suffix, and
    # every env is now "beamng"), which left runs with no plot beside their model
    # and silently overwrote the plot of whatever model owned that stem.
    plot_path = f"{os.path.splitext(save_path)[0]}_training.png"
    start_episode = 0

    if os.path.exists(save_path):
        if request.reset_existing:
            os.remove(save_path)
        else:
            agent.load(save_path)
            start_episode = getattr(agent, "episode", 0)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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
        "plot_path": plot_path,
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
    env = env_info["factory"](
        **_beamng_kwargs(
            request.beamng or BeamNGOptions(), request.algo_name, with_random_path=False
        )
    )

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


def _benchmark_env(request: BenchmarkRequest, algo_name: str | None = None):
    """Resolve (env_factory, env_metadata) for a benchmark run.

    Benchmarks call ``env_factory()`` with no arguments, so the request's options
    are baked into the factory and the observation/action sizes are written into
    the metadata the benchmark uses to build agents — mirroring ``build_agent``.
    The one factory also feeds ``evaluate_policy``, so the training-only options
    (``random_path``, ``dense_episodes``) are deliberately left out: eval must stay
    sparse and on a fixed path (see ``run_evaluate``). Pass ``algo_name=None`` to
    leave the env's output axis at the factory default.
    """
    env_info = registry.get_environment(request.env_name)
    factory = env_info["factory"]
    metadata = dict(env_info["metadata"])
    if not request.env_name.startswith("beamng"):
        return factory, metadata

    beamng = request.beamng or BeamNGOptions()
    kwargs = _beamng_kwargs(beamng, algo_name, with_random_path=False)
    metadata["n_states"] = beamng_spec.obs_size(
        beamng.sensor, beamng.trajectory_hints, beamng.body_orientation, beamng.road_info
    )
    if algo_name is not None:
        metadata["n_actions"] = beamng_spec.action_size(beamng_spec.output_for_algo(algo_name))
    return (lambda: factory(**kwargs)), metadata


def _benchmark_agent_params(algo_name: str, metadata: dict) -> dict:
    """Agent params for a benchmark: algorithm defaults + env-driven typing.

    Benchmarks instantiate ``agent_cls(**params)`` themselves, taking
    ``n_states`` / ``n_actions`` from ``env_metadata``. This mirrors the rest
    of ``build_agent``: ``state_type`` is passed only to constructors that
    accept it, and a discrete env fixes the action count over any
    continuous-control default (e.g. TD3's ``n_actions=2``).
    """
    algo_info = registry.get_algorithm(algo_name)
    params = dict(algo_info["default_config"])
    if "state_type" in inspect.signature(algo_info["class"]).parameters:
        params["state_type"] = metadata.get("state_type", "continuous")
    if metadata.get("state_type") == "discrete":
        params["n_actions"] = metadata.get("n_actions")
    return params


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
    env_factory, metadata = _benchmark_env(request, request.algo_name)
    config = {
        "agent_params": _benchmark_agent_params(request.algo_name, metadata),
        "env_metadata": metadata,
        "threshold": request.reward_threshold,
        **common,
    }
    seeds = common["seeds"]
    if len(seeds) > 1:
        results = bench.run_multi(algo_info["class"], env_factory, config)
        bench.export_multi(results, request.algo_name, request.env_name)
        return {"status": "ok", "summary": format_aggregate(results)}
    config["seed"] = seeds[0]
    results = bench.run(algo_info["class"], env_factory, config)
    bench.export(results, request.algo_name, request.env_name)
    return {"status": "ok", "report": bench.report(results)}


def _run_comparison(bench, request: BenchmarkRequest, common: dict) -> dict[str, object]:
    algos = request.algos
    # Every variant shares this one env factory, so BeamNG keeps the default
    # reward mode here (per-algo reward modes would need per-variant factories).
    env_factory, metadata = _benchmark_env(request)
    config = {
        "env_metadata": metadata,
        "threshold": request.reward_threshold,
        "window": 100,
        "variants": [
            {"name": algo, "algo": algo, "agent_params": _benchmark_agent_params(algo, metadata)}
            for algo in algos
        ],
        **common,
    }
    results = bench.run(None, env_factory, config)
    lines = []
    for label, data in results["variants"].items():
        stat = data["aggregate"].get("eval_mean_reward", {})
        lines.append(f"  {label}: eval reward {stat.get('mean', 0)} +/- {stat.get('std', 0)}")
    bench.export(results, "+".join(algos), request.env_name)
    return {"status": "ok", "summary": "\n".join(lines)}


def _run_gridsearch(bench, request: BenchmarkRequest, common: dict) -> dict[str, object]:
    algo_info = registry.get_algorithm(request.algo_name)
    env_factory, metadata = _benchmark_env(request, request.algo_name)
    config = {
        "agent_params": _benchmark_agent_params(request.algo_name, metadata),
        "env_metadata": metadata,
        "param_grid": request.param_grid,
        **common,
    }
    results = bench.run(algo_info["class"], env_factory, config)
    bench.export(results, request.algo_name, request.env_name)
    return {"status": "ok", "report": bench.report(results)}


# --------------------------------------------------------------------------- #
# Human play
# --------------------------------------------------------------------------- #
def run_human_play(request: HumanPlayRequest) -> None:
    """Drive manually with the chosen sensor's observation shown live.

    One env and one loop for every sensor: ``human_play`` adapts its readout to the
    sensor (per-cell LiDAR bins plus filtering diagnostics, or the dashcam frame as
    ASCII art). The output axis is irrelevant here — the human is the policy.
    """
    env = None
    try:
        env = registry.get_environment("beamng")["factory"](
            map_name=request.map_name,
            sensor=request.sensor,
            # A game track is one fixed line, so there is no random path to deal.
            random_path=request.random_path and not request.track,
            track=request.track or None,
        )
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
def build_multi_session(
    specs: list[dict], map_name: str, random_path: bool = False, track: str = ""
):
    """Create the BeamNGMultiEnv and an agent per spec.

    Each spec carries its own ``sensor``; the agent is sized to that sensor's
    observation length (``slot_n_states``) and to the action count the spec's
    algorithm implies. Returns (env, slots).

    ``track`` names one of the game's race tracks; every vehicle then trains on
    that shared line from a starting grid instead of getting its own path.
    """
    env = BeamNGMultiEnv(
        slots=[],
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        map_name=map_name,
        random_path=random_path and not track,
        track=track or None,
    )

    enriched = []
    for spec in specs:
        algo_info = registry.get_algorithm(spec["algo"])
        cls = algo_info["class"]
        cfg = dict(algo_info["default_config"])
        cfg["n_states"] = slot_n_states(
            spec.get("sensor", beamng_spec.DEFAULT_SENSOR),
            spec.get("trajectory_hints", 0),
            spec.get("body_orientation", False),
            spec.get("road_info", False),
        )
        cfg["n_actions"] = beamng_spec.action_size(beamng_spec.output_for_algo(spec["algo"]))
        cfg.pop("state_type", None)
        enriched.append({**spec, "agent": cls(**cfg)})

    slots = build_slots(enriched)
    env.slots = slots
    return env, slots


RACE_COLORS: tuple[str, ...] = ("Red", "Blue", "Yellow", "Green", "White", "Black")

# Race checkpoints (race-training writes here) live beside the multi-agent ones.
RACE_OUTPUT_DIR = os.path.join("outputs", "races")


def _validate_course(request: CourseRequest) -> None:
    """Reject a race that cannot be run, with a reason the user can act on."""
    if request.laps != 1:
        raise ValueError(
            f"laps={request.laps} is not supported yet: the generated paths are open "
            "roads, so a second lap would mean driving back to the start. Use laps=1."
        )
    if len(request.racers) < 2:
        raise ValueError("a race needs at least two entrants")
    if sum(1 for r in request.racers if r.human) > 1:
        raise ValueError("only one human can race at a time — there is one keyboard")
    for racer in request.racers:
        if racer.human:
            continue
        if not racer.algo:
            raise ValueError("every non-human entrant needs an algorithm")
        if not racer.model_path:
            raise ValueError(f"entrant '{racer.algo}' needs a checkpoint to race")
        if not os.path.exists(racer.model_path):
            raise FileNotFoundError(racer.model_path)


def build_course_session(request: CourseRequest):
    """Build the race env and load each entrant's checkpoint. Returns (env, slots).

    A human entrant needs realtime pacing — nobody can drive in lockstep — so the
    env's mode is derived from the field rather than asked for separately.
    """
    from core.race_runner import RaceRunner  # noqa: F401 — import symmetry with callers
    from environments.beamng_race import BeamNGRaceEnv, build_race_slots

    specs: list[dict] = []
    for i, racer in enumerate(request.racers):
        color = racer.color or RACE_COLORS[i % len(RACE_COLORS)]
        if racer.human:
            specs.append({"human": True, "color": color})
            continue

        algo_info = registry.get_algorithm(racer.algo)
        cfg = dict(algo_info["default_config"])
        cfg["n_states"] = beamng_spec.obs_size(
            racer.sensor, racer.trajectory_hints, racer.body_orientation, racer.road_info
        )
        cfg["n_actions"] = beamng_spec.action_size(beamng_spec.output_for_algo(racer.algo))
        cfg.pop("state_type", None)
        agent = algo_info["class"](**cfg)
        agent.load(racer.model_path)

        # Race-training writes to its own file, so an exhibition race can never
        # damage the checkpoint that was handed to it.
        save_path = os.path.join(
            RACE_OUTPUT_DIR, f"{racer.algo}_{racer.sensor}_race{i}.pth"
        )
        specs.append(
            {
                "algo": racer.algo,
                "agent": agent,
                "color": color,
                "save_path": save_path,
                "sensor": racer.sensor,
                "trajectory_hints": racer.trajectory_hints,
                "body_orientation": racer.body_orientation,
                "road_info": racer.road_info,
            }
        )

    has_human = any(r.human for r in request.racers)
    env = BeamNGRaceEnv(
        slots=[],
        beamng_home=BEAMNG_HOME,
        beamng_user=BEAMNG_USER,
        headless=HEADLESS,
        map_name=request.map_name,
        path_idx=request.path_idx,
        laps=request.laps,
        realtime=has_human,
        track=request.track or None,
    )
    slots = build_race_slots(specs)
    env.slots = slots
    return env, slots


def run_course(request: CourseRequest) -> dict[str, object]:
    """Race the field on one shared track and report the outcome."""
    _validate_course(request)

    from core.race_runner import RaceRunner

    env, slots = build_course_session(request)
    if request.learning:
        os.makedirs(RACE_OUTPUT_DIR, exist_ok=True)

    runner = RaceRunner()
    try:
        outcome = runner.run(env, races=request.races, learning=request.learning)
    finally:
        env.close()

    return {
        "status": "ok",
        "learning": request.learning,
        "entrants": [s.name for s in slots],
        **outcome,
    }


def run_multi_train(request: MultiTrainRequest) -> dict[str, object]:
    env, slots = build_multi_session(
        request.specs, request.map_name, request.random_path, request.track
    )

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
