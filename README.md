# Taxi-v3 Q-Learning Agent

Reinforcement learning agent trained on the [Gymnasium Taxi-v3](https://gymnasium.farama.org/environments/toy_text/taxi/) environment using Q-Learning.

---

## Requirements

- Python 3.10+

---

## Installation

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

**Windows:**
```bash
venv\Scripts\activate
```

**Linux / macOS:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

### Start the program

```bash
python main.py
```

You will be prompted to choose a mode:

```
1. User mode        — tune algorithm parameters
2. Time-limited mode — optimized params, fixed time budget
```

---

## Modes

### Mode 1 — User mode

Lets you manually set every hyperparameter before training.
Press **Enter** to keep the default value shown in brackets.

| Parameter | Description | Default |
|---|---|---|
| Learning rate (alpha) | Step size for Q-table updates | `0.1` |
| Discount factor (gamma) | Weight of future rewards | `0.99` |
| Initial epsilon | Starting exploration rate | `1.0` |
| Epsilon min | Minimum exploration rate | `0.01` |
| Epsilon decay | Multiplicative decay per episode | `0.995` |
| Training episodes | Number of episodes to train | `10000` |
| Testing episodes | Number of episodes to evaluate | `10` |

### Mode 2 — Time-limited mode

Uses pre-tuned optimized parameters. You only provide a time budget and episode counts.
Training stops as soon as the time limit is reached, even if not all episodes are done.

| Prompt | Description | Default |
|---|---|---|
| Time limit (seconds) | Max training duration | `60` |
| Max training episodes | Upper bound on training | `50000` |
| Testing episodes | Number of episodes to evaluate | `10` |

---

## Output files

| File | Description |
|---|---|
| `q_table.npy` | Saved Q-table after training |
| `training_results.png` | Plot of rewards and steps over training |

---

## Run training or evaluation standalone

Train only:
```bash
python train.py
```

Evaluate a previously saved Q-table:
```bash
python evaluate.py
```
