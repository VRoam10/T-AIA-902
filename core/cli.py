"""Interactive CLI menu for the RL pipeline."""

import os

from core.registry import registry
from core.runner import PipelineRunner


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


def _ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            val = int(raw)
            if val > 0:
                return val
            print("  Please enter a positive integer.")
        except ValueError:
            print("  Invalid input.")


def _ask_float(prompt: str, default: float) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if raw == "":
            return default
        try:
            return float(raw)
        except ValueError:
            print("  Invalid input.")


def _build_agent(algo_info: dict, env_info: dict):
    """Instantiate an agent from registry info, prompting for hyperparams."""
    cls = algo_info["class"]
    defaults = dict(algo_info["default_config"])
    meta = env_info["metadata"]

    # Inject n_states / n_actions from env metadata
    defaults["n_states"] = meta.get("n_states", 5)
    defaults["n_actions"] = meta.get("n_actions", 6)

    print("\nHyperparameters (press Enter for default):")
    params = {}
    for key, default_val in defaults.items():
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

    agent = _build_agent(algo_info, env_info)
    env = env_info["factory"]()

    n_episodes = _ask_int("\nNumber of episodes", 500)
    save_path = input(f"Save model path [outputs/{algo_name}_{env_name}.pth]: ").strip()
    if not save_path:
        save_path = f"outputs/{algo_name}_{env_name}.pth"
    plot_path = f"outputs/{algo_name}_{env_name}_training.png"

    # Resume from checkpoint if it exists
    if os.path.exists(save_path):
        print(f"Found existing model at '{save_path}', resuming.")
        agent.load(save_path)

    os.makedirs("outputs", exist_ok=True)

    runner = PipelineRunner()
    print(f"\n--- Training {algo_name} on {env_name} ({n_episodes} episodes) ---\n")
    runner.train(
        agent,
        env,
        n_episodes=n_episodes,
        save_path=save_path,
        plot_path=plot_path,
    )
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

    agent = _build_agent(algo_info, env_info)
    env = env_info["factory"]()

    model_path = input(f"Model path [outputs/{algo_name}_{env_name}.pth]: ").strip()
    if not model_path:
        model_path = f"outputs/{algo_name}_{env_name}.pth"

    if not os.path.exists(model_path):
        print(f"Model not found at '{model_path}'.")
        env.close()
        return

    agent.load(model_path)
    n_episodes = _ask_int("Number of evaluation episodes", 10)

    runner = PipelineRunner()
    print(f"\n--- Evaluating {algo_name} on {env_name} ({n_episodes} episodes) ---\n")
    runner.evaluate(agent, env, n_episodes=n_episodes)
    env.close()


def _benchmark_menu():
    print("\n--- Run a Benchmark ---")

    benchmarks = registry.list_benchmarks()
    if not benchmarks:
        print("No benchmarks registered.")
        return
    print("\nAvailable benchmarks:")
    bench_name = _pick(benchmarks, "Benchmark")

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


def _human_play_menu():
    print("\n--- Human Play (BeamNG) ---")
    envs = registry.list_environments()
    if "beamng" not in envs:
        print("BeamNG environment not registered.")
        return

    env_info = registry.get_environment("beamng")
    env = env_info["factory"]()

    print("Launching BeamNG for human play...")
    env.human_play()

    input("\nPress Enter when done playing...")
    env.close()


def main_menu():
    """Main interactive CLI loop."""
    # Trigger auto-registration by importing packages
    import algorithms  # noqa: F401
    import benchmarks  # noqa: F401
    import environments  # noqa: F401

    while True:
        print("\n" + "=" * 50)
        print("   RL Pipeline")
        print("=" * 50)
        print("1. Train an agent")
        print("2. Evaluate an agent")
        print("3. Run a benchmark")
        print("4. Human play (BeamNG)")
        print("5. Quit")

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
            print("Bye!")
            break
        else:
            print("  Invalid choice.")
