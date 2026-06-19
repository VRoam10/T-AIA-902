"""Interactive CLI menu for the RL pipeline."""

import os

import algorithms  # noqa: F401 — triggers registry auto-registration
import benchmarks  # noqa: F401 — triggers registry auto-registration
import environments  # noqa: F401 — triggers registry auto-registration
from config import BEAMNG_HOME, BEAMNG_USER, HEADLESS
from core.multi_runner import MultiAgentRunner
from core.registry import registry
from core.runner import PipelineRunner
from environments.beamng_multi import BeamNGMultiEnv, build_slots, slot_n_states


def format_trajectory_summary(mt) -> str:
    """One-line summary of a MapTrajectories: path count + per-path sources."""
    n = len(mt.paths)
    sources = ", ".join(p.source for p in mt.paths)
    return f"{mt.map_name}: {n} path(s) [{sources}]"


def _pick(options: list[str], prompt: str = "Select") -> str:
    """Display numbered options and return the chosen one."""
    for i, name in enumerate(options, 1):
        print(f"  {i}. {name}")
    while True:
        raw = input(f"\n{prompt} (number): ").strip()
        try:
            idx = int(raw) - 1
            if 0 <= idx < len(options):
                return options[idx]
        except ValueError:
            pass
        print("  Invalid choice, try again.")


def _ask_int(prompt: str, default: int, min_val: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val >= min_val:
                return val
            print(f"  Please enter an integer >= {min_val}.")
        except ValueError:
            print("  Invalid input.")


def _ask_bool(prompt: str, default: bool = False) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


_BEAMNG_MAPS = ["gridmap_v2", "italy", "west_coast_usa"]

_BEAMNG_VEHICLES = {
    "taxi": "Burnside (Taxi)",
    "gavril_t_series": "Gavril T-Series",
    "ibishu_pigeon": "Ibishu Pigeon",
    "gavril_d_series": "Gavril D-Series",
}

_MULTI_ALGOS = ["dqn", "ddpg", "td3"]

# Multi-agent checkpoints + plots live in their own subfolder.
_MULTI_OUTPUT_DIR = os.path.join("outputs", "multi-agents")


def _pick_beamng_options() -> tuple[str, str]:
    """Prompt for map and vehicle when launching a BeamNG environment."""
    print("\nAvailable maps:")
    map_name = _pick(_BEAMNG_MAPS, "Map")
    print(f"\n  Selected map : {map_name}")
    cache_path = os.path.join("outputs", "trajectories", f"{map_name}.json")
    if not os.path.exists(cache_path):
        print(
            f"  Note: no cached trajectory for '{map_name}'. It will be generated on first launch."
        )

    vehicle_labels = list(_BEAMNG_VEHICLES.values())
    vehicle_keys = list(_BEAMNG_VEHICLES.keys())
    print("\nAvailable vehicles:")
    vehicle_label = _pick(vehicle_labels, "Vehicle")
    vehicle_id = vehicle_keys[vehicle_labels.index(vehicle_label)]
    print(f"  Selected vehicle: {vehicle_label}")

    return map_name, vehicle_id


def _ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Invalid input.")


def _build_agent(algo_info: dict, env_info: dict, prompt_params: bool = True):
    """Instantiate an agent from registry info.

    If prompt_params is False, use default config without asking (for eval).
    """
    cls = algo_info["class"]
    defaults = dict(algo_info["default_config"])
    meta = env_info["metadata"]

    # Inject environment metadata into agent constructor params
    defaults["n_states"] = meta.get("n_states", 5)
    defaults["n_actions"] = meta.get("n_actions", defaults.get("n_actions", 6))
    defaults["state_type"] = meta.get("state_type", "continuous")

    if not prompt_params:
        defaults.pop("state_type", None)
        return cls(**defaults)

    print("\nHyperparameters (press Enter for default):")
    params = {}
    for key, default_val in defaults.items():
        if key == "state_type":
            continue
        if key in ("n_states", "n_actions"):
            params[key] = default_val
            continue
        if isinstance(default_val, int):
            params[key] = _ask_int(f"  {key}", default_val)
        elif isinstance(default_val, float):
            params[key] = _ask_float(f"  {key}", default_val)
        else:
            params[key] = default_val

    return cls(**params)


def _train_menu():
    print("\n--- Train an Agent ---")

    algos = registry.list_algorithms()
    if not algos:
        print("No algorithms registered.")
        return
    print("\nAvailable algorithms:")
    algo_name = _pick(algos, "Algorithm")

    envs = registry.compatible_environments(algo_name)
    if not envs:
        print("No compatible environments for this algorithm.")
        return
    print("\nAvailable environments:")
    env_name = _pick(envs, "Environment")

    algo_info = registry.get_algorithm(algo_name)
    env_info = registry.get_environment(env_name)

    beamng_kwargs = {}
    trajectory_hints = 0
    body_orientation = False
    wheel_terrain = False
    if env_name.startswith("beamng"):
        map_name, vehicle_id = _pick_beamng_options()
        trajectory_hints = _ask_int(
            "\nCheckpoint hints (waypoints ahead in obs, 0 = none)", 0, min_val=0
        )
        body_orientation = _ask_bool("Include body orientation (pitch + roll) in obs?")
        wheel_terrain = _ask_bool("Include per-wheel road position in obs?")
        random_path = _ask_bool("Randomize path each episode?")
        beamng_kwargs = {
            "map_name": map_name,
            "vehicle_id": vehicle_id,
            "trajectory_hints": trajectory_hints,
            "body_orientation": body_orientation,
            "wheel_terrain": wheel_terrain,
            "random_path": random_path,
        }

    # Adjust n_states for the chosen options before building the agent
    extra_states = (
        trajectory_hints * 2 + (2 if body_orientation else 0) + (2 if wheel_terrain else 0)
    )
    env_meta = {
        **env_info["metadata"],
        "n_states": env_info["metadata"]["n_states"] + extra_states,
    }
    agent = _build_agent(algo_info, {**env_info, "metadata": env_meta})

    # Continuous-action algorithms get their own reward mode
    reward_mode = algo_name if algo_name in ("ddpg", "td3") else "default"
    env = env_info["factory"](reward_mode=reward_mode, **beamng_kwargs)

    n_episodes = _ask_int("\nNumber of episodes", 500)
    save_path = input(f"Save model path [outputs/{algo_name}_{env_name}.pth]: ").strip()
    if not save_path:
        save_path = f"outputs/{algo_name}_{env_name}.pth"
    plot_path = f"outputs/{algo_name}_{env_name}_training.png"

    start_episode = 0

    # Resume or reset if checkpoint exists
    if os.path.exists(save_path):
        prev_ep = "?"
        # Peek at saved episode count
        try:
            import torch

            ckpt = torch.load(save_path, map_location="cpu")
            prev_ep = ckpt.get("episode", "?")
        except Exception:
            pass
        print(f"\nFound existing model at '{save_path}' (episode {prev_ep}).")
        choice = input("  [C]ontinue training  /  [R]eset from scratch? [C/R]: ").strip().lower()
        if choice == "r":
            os.remove(save_path)
            print("  Checkpoint deleted — starting fresh.")
        else:
            agent.load(save_path)
            start_episode = getattr(agent, "episode", 0)
            print(f"  Resuming from episode {start_episode}.")

    os.makedirs("outputs", exist_ok=True)

    total = start_episode + n_episodes
    runner = PipelineRunner()
    print(
        f"\n--- Training {algo_name} on {env_name} (episodes {start_episode + 1} -> {total}) ---\n"
    )
    try:
        runner.train(
            agent,
            env,
            n_episodes=n_episodes,
            save_path=save_path,
            plot_path=plot_path,
            start_episode=start_episode,
        )
    finally:
        env.close()


def _eval_menu():
    print("\n--- Evaluate an Agent ---")

    algos = registry.list_algorithms()
    if not algos:
        print("No algorithms registered.")
        return
    print("\nAvailable algorithms:")
    algo_name = _pick(algos, "Algorithm")

    envs = registry.compatible_environments(algo_name)
    if not envs:
        print("No compatible environments.")
        return
    print("\nAvailable environments:")
    env_name = _pick(envs, "Environment")

    algo_info = registry.get_algorithm(algo_name)
    env_info = registry.get_environment(env_name)

    model_path = input(f"\nModel path [outputs/{algo_name}_{env_name}.pth]: ").strip()
    if not model_path:
        model_path = f"outputs/{algo_name}_{env_name}.pth"

    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'.")
        return

    beamng_kwargs = {}
    trajectory_hints = 0
    body_orientation = False
    wheel_terrain = False
    if env_name.startswith("beamng"):
        map_name, vehicle_id = _pick_beamng_options()
        trajectory_hints = _ask_int(
            "\nCheckpoint hints (must match the trained model)", 0, min_val=0
        )
        body_orientation = _ask_bool("Body orientation in obs? (must match the trained model)")
        wheel_terrain = _ask_bool("Per-wheel road position in obs? (must match the trained model)")
        beamng_kwargs = {
            "map_name": map_name,
            "vehicle_id": vehicle_id,
            "trajectory_hints": trajectory_hints,
            "body_orientation": body_orientation,
            "wheel_terrain": wheel_terrain,
        }

    extra_states = (
        trajectory_hints * 2 + (2 if body_orientation else 0) + (2 if wheel_terrain else 0)
    )
    env_meta = {
        **env_info["metadata"],
        "n_states": env_info["metadata"]["n_states"] + extra_states,
    }
    agent = _build_agent(algo_info, {**env_info, "metadata": env_meta}, prompt_params=False)
    agent.load(model_path)

    env = env_info["factory"](**beamng_kwargs)

    n_episodes = _ask_int("Number of evaluation episodes", 10)

    runner = PipelineRunner()
    print(f"\n--- Evaluating {algo_name} on {env_name} ({n_episodes} episodes) ---\n")
    try:
        runner.evaluate(agent, env, n_episodes=n_episodes)
    finally:
        env.close()


def _benchmark_menu():
    print("\n--- Run a Benchmark ---")

    bench_names = registry.list_benchmarks()
    if not bench_names:
        print("No benchmarks registered.")
        return
    print("\nAvailable benchmarks:")
    bench_name = _pick(bench_names, "Benchmark")

    algos = registry.list_algorithms()
    print("\nAvailable algorithms:")
    algo_name = _pick(algos, "Algorithm")

    envs = registry.compatible_environments(algo_name)
    print("\nAvailable environments:")
    env_name = _pick(envs, "Environment")

    algo_info = registry.get_algorithm(algo_name)
    env_info = registry.get_environment(env_name)
    bench_info = registry.get_benchmark(bench_name)

    bench = bench_info["class"]()

    config = {
        "agent_params": algo_info["default_config"],
        "env_metadata": env_info["metadata"],
        "max_episodes": _ask_int("Max episodes", 2000),
        "threshold": _ask_float("Reward threshold", 7.0),
    }

    print(f"\n--- Running {bench_name}: {algo_name} + {env_name} ---\n")
    results = bench.run(algo_info["class"], env_info["factory"], config)
    print("\n" + bench.report(results))
    bench.export(results, algo_name, env_name)


def _human_play_menu():
    print("\n--- Human Play (BeamNG) ---")
    envs = registry.list_environments()
    if "beamng" not in envs:
        print("BeamNG environment not registered.")
        return

    sensor_options = ["None", "LiDAR"]
    if "beamng_camera" in envs:
        sensor_options.append("Camera")

    map_name, vehicle_id = _pick_beamng_options()

    print("\nShow sensor during play?")
    sensor = _pick(sensor_options, "Sensor")

    env = None
    print("Launching BeamNG for human play...")
    try:
        if sensor == "Camera":
            env = registry.get_environment("beamng_camera")["factory"](
                map_name=map_name, vehicle_id=vehicle_id
            )
        else:
            env = registry.get_environment("beamng")["factory"](
                map_name=map_name, vehicle_id=vehicle_id
            )

        if sensor == "LiDAR":
            env.human_play_lidar()
        elif sensor == "Camera":
            env.human_play_camera()
        else:
            env.human_play()
    finally:
        if env is not None:
            env.close(kill_sim=False)


def _trajectory_menu():
    """Pre-warm the trajectory cache for one or all BeamNG maps."""
    print("\n--- Generate Trajectories ---")
    print("This will launch BeamNG and probe each map's road network.")
    print("The result is cached in outputs/trajectories/<map>.json.\n")

    options = _BEAMNG_MAPS + ["all"]
    choice = _pick(options, "Map")

    targets = _BEAMNG_MAPS if choice == "all" else [choice]

    from environments.beamng import BeamNGDrivingEnv

    for map_name in targets:
        print(f"\n>>> Generating trajectory for '{map_name}' ...")
        cache_path = os.path.join("outputs", "trajectories", f"{map_name}.json")
        if os.path.exists(cache_path):
            ans = (
                input(f"    Cache already exists at {cache_path}. Overwrite? [y/N]: ")
                .strip()
                .lower()
            )
            if ans != "y":
                print("    Skipped.")
                continue
            os.remove(cache_path)

        env = BeamNGDrivingEnv(
            beamng_home=BEAMNG_HOME,
            beamng_user=BEAMNG_USER,
            headless=HEADLESS,
            map_name=map_name,
        )
        try:
            env.reset()
            from core.trajectory import load_or_generate

            mt = load_or_generate(map_name, bng=None)
            print(f"    Done. {format_trajectory_summary(mt)}")
        finally:
            env.close()


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


def _multi_train_menu():
    print("\n--- Multi-Agent Training (BeamNG) ---")
    print("\nAvailable maps:")
    map_name = _pick(_BEAMNG_MAPS, "Map")
    random_path = _ask_bool("Randomize path each episode (deals distinct paths per vehicle)?")

    vehicle_keys = list(_BEAMNG_VEHICLES.keys())
    vehicle_labels = list(_BEAMNG_VEHICLES.values())
    colors = ["Yellow", "Red", "Blue", "Green", "Orange", "White", "Black"]

    specs = []
    while True:
        print(f"\n--- Vehicle {len(specs)} ---")
        print("Algorithm:")
        algo = _pick(_MULTI_ALGOS, "Algorithm")
        # Environments compatible with this algorithm (BeamNG only).
        env_options = [e for e in registry.compatible_environments(algo) if e.startswith("beamng")]
        print("Environment:")
        env_name = _pick(env_options, "Environment")
        print("Vehicle model:")
        vlabel = _pick(vehicle_labels, "Vehicle")
        vehicle_id = vehicle_keys[vehicle_labels.index(vlabel)]
        color = colors[len(specs) % len(colors)]
        hints = _ask_int("Checkpoint hints (waypoints ahead in obs, 0 = none)", 0, min_val=0)
        body_orientation = _ask_bool("Include body orientation (pitch + roll) in obs?")
        wheel_terrain = _ask_bool("Include per-wheel road position in obs?")
        default_path = os.path.join(_MULTI_OUTPUT_DIR, f"{algo}_{env_name}_{len(specs)}.pth")
        save_path = input(f"  Model save path [{default_path}]: ").strip() or default_path
        specs.append(
            {
                "algo": algo,
                "env": env_name,
                "vehicle_id": vehicle_id,
                "color": color,
                "save_path": save_path,
                "trajectory_hints": hints,
                "body_orientation": body_orientation,
                "wheel_terrain": wheel_terrain,
            }
        )
        more = input("\nAdd another vehicle? [y/N]: ").strip().lower()
        if more != "y":
            break

    if not specs:
        print("No vehicles configured.")
        return

    n_episodes = _ask_int("\nEpisodes per agent", 500)
    minutes = _ask_float("Time limit (minutes, 0 = none)", 0.0)
    time_limit = minutes * 60.0 if minutes > 0 else None

    env, slots = build_multi_session(specs, map_name, random_path)
    for slot in slots:
        if os.path.exists(slot.save_path):
            choice = (
                input(f"  [{slot.name}] '{slot.save_path}' exists. [C]ontinue / [R]eset? [C/R]: ")
                .strip()
                .lower()
            )
            if choice == "r":
                os.remove(slot.save_path)
            else:
                slot.agent.load(slot.save_path)
                slot.episode = getattr(slot.agent, "episode", 0)

    os.makedirs(_MULTI_OUTPUT_DIR, exist_ok=True)
    runner = MultiAgentRunner()
    print(f"\n--- Training {len(slots)} agents on {map_name} ---\n")
    try:
        runner.train(env, n_episodes=n_episodes, time_limit=time_limit)
    finally:
        env.close()


def main_menu():
    """Main interactive CLI loop."""
    while True:
        print("\n" + "=" * 50)
        print("   RL Pipeline")
        print("=" * 50)
        print("1. Train an agent")
        print("2. Evaluate an agent")
        print("3. Run a benchmark")
        print("4. Human play (BeamNG)")
        print("5. Generate trajectories (BeamNG)")
        print("6. Multi-agent training (BeamNG)")
        print("7. Quit")

        choice = input("\nSelect: ").strip()

        if choice == "1":
            _train_menu()
        elif choice == "2":
            _eval_menu()
        elif choice == "3":
            _benchmark_menu()
        elif choice == "4":
            _human_play_menu()
        elif choice == "5":
            _trajectory_menu()
        elif choice == "6":
            _multi_train_menu()
        elif choice == "7":
            print("Bye!")
            break
        else:
            print("  Invalid choice.")
