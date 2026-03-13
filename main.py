from evaluate import evaluate
from train import OPTIMIZED_PARAMS, train


def ask_int(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            value = int(raw)
            if value > 0:
                return value
            print("  Please enter a positive integer.")
        except ValueError:
            print("  Invalid input, please enter an integer.")


def ask_float(prompt: str, default: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    while True:
        raw = input(f"{prompt} [default: {default}]: ").strip()
        if raw == "":
            return default
        try:
            value = float(raw)
            if min_val <= value <= max_val:
                return value
            print(f"  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  Invalid input, please enter a number.")


def user_mode():
    print("\n--- User Mode: Tune Hyperparameters ---")
    print("Press Enter to keep the default value.\n")

    learning_rate  = ask_float("Learning rate (alpha)", default=0.1)
    discount_factor = ask_float("Discount factor (gamma)", default=0.99)
    epsilon        = ask_float("Initial epsilon", default=1.0)
    epsilon_min    = ask_float("Epsilon min", default=0.01)
    epsilon_decay  = ask_float("Epsilon decay", default=0.995)

    print()
    n_train = ask_int("Number of training episodes", default=10000)
    n_test  = ask_int("Number of testing episodes",  default=10)

    print("\n--- Training ---")
    train(
        n_episodes=n_train,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
    )

    print("\n--- Evaluation ---")
    evaluate(n_episodes=n_test)


def time_limited_mode():
    print("\n--- Time-Limited Mode: Optimized Parameters ---")
    print("Using pre-tuned hyperparameters for best performance.\n")

    for k, v in OPTIMIZED_PARAMS.items():
        print(f"  {k}: {v}")

    print()
    time_limit = ask_float("Training time limit in seconds", default=60.0, min_val=1.0, max_val=3600.0)
    n_train    = ask_int("Max training episodes (stops earlier if time runs out)", default=50000)
    n_test     = ask_int("Number of testing episodes", default=10)

    print("\n--- Training ---")
    train(
        n_episodes=n_train,
        time_limit=time_limit,
        **OPTIMIZED_PARAMS,
    )

    print("\n--- Evaluation ---")
    evaluate(n_episodes=n_test)


def main():
    print("=" * 45)
    print("       Taxi-v3 Q-Learning Agent")
    print("=" * 45)
    print("\nSelect a mode:")
    print("  1. User mode       — tune algorithm parameters")
    print("  2. Time-limited mode — optimized params, fixed time budget")

    while True:
        choice = input("\nEnter 1 or 2: ").strip()
        if choice == "1":
            user_mode()
            break
        elif choice == "2":
            time_limited_mode()
            break
        else:
            print("  Please enter 1 or 2.")


if __name__ == "__main__":
    main()
