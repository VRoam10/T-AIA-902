# RL Pipeline

Pipeline de Reinforcement Learning modulaire pour entrainer et evaluer des agents sur differents environnements (Taxi-v3, BeamNG.drive).

Ajouter un nouvel algorithme = **1 fichier + 1 ligne de registration**. Il apparait automatiquement dans le menu.

---

## Requirements

- Python 3.10+
- BeamNG.drive (optionnel, uniquement pour l'environnement BeamNG)

---

## Installation

### 1. Creer un environnement virtuel

```bash
python -m venv .venv
```

### 2. Activer l'environnement virtuel

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 3. Installer les dependances

```bash
pip install -r requirements.txt
```

Pour le support GPU (CUDA) avec PyTorch :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### 4. Configurer l'environnement

Copier le template et remplir avec vos chemins :

```bash
cp .env.template .env
```

Editer `.env` :

```
BEAMNG_HOME=C:\chemin\vers\BeamNG.drive
BEAMNG_USER=C:\chemin\vers\BeamNG\user\folder
```

> Ces variables ne sont necessaires que si vous utilisez l'environnement BeamNG.

---

## Utilisation

```bash
python main.py
```

Le menu interactif s'affiche :

```
==================================================
   RL Pipeline
==================================================
1. Train an agent
2. Evaluate an agent
3. Run a benchmark
4. Human play (BeamNG)
5. Quit
```

### 1. Train an agent

- Choisir un algorithme (`q_learning`, `dqn`, ...)
- Choisir un environnement compatible
- Ajuster les hyperparametres (ou garder les defaults)
- L'entrainement demarre, les modeles et plots sont sauvegardes dans `outputs/`

### 2. Evaluate an agent

- Charger un modele sauvegarde
- Lancer N episodes en mode exploitation (epsilon=0)

### 3. Run a benchmark

- Lancer un benchmark (ex: `convergence`) sur une combinaison algo + env
- Les resultats sont affiches a la fin

### 4. Human play (BeamNG)

- Conduire manuellement dans BeamNG pour tester le scenario

---

## Structure du projet

```
testRomain/
├── main.py                  # Point d'entree -> menu interactif
├── config.py                # Charge les variables depuis .env
├── .env                     # Variables d'environnement (non versionne)
├── .env.template            # Template a copier
│
├── core/
│   ├── base_agent.py        # Classe abstraite BaseAgent
│   ├── base_benchmark.py    # Classe abstraite BaseBenchmark
│   ├── registry.py          # Registre central (algos, envs, benchmarks)
│   ├── runner.py            # Boucle train/eval generique
│   └── cli.py               # Menu interactif
│
├── algorithms/
│   ├── __init__.py           # Registration des algorithmes
│   ├── q_learning.py         # Q-Learning (Taxi-v3)
│   └── dqn.py                # Double DQN (BeamNG, Taxi)
│
├── environments/
│   ├── __init__.py           # Registration des environnements
│   ├── taxi.py               # Factory Taxi-v3
│   └── beamng.py             # BeamNG.drive wrapper
│
├── benchmarks/
│   ├── __init__.py           # Registration des benchmarks
│   └── convergence.py        # Benchmark de convergence
│
└── outputs/                  # Modeles et plots (non versionne)
```

---

## Ajouter un nouvel algorithme

### Etape 1 : Creer le fichier

Creer `algorithms/mon_algo.py` :

```python
from core.base_agent import BaseAgent
import numpy as np


class MonAlgoAgent(BaseAgent):
    def __init__(self, n_states, n_actions, lr=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        # ... initialiser ...

    def select_action(self, state) -> int:
        # Choisir une action (epsilon-greedy ou autre)
        ...

    def update(self, state, action, reward, next_state, done):
        # Mettre a jour l'agent avec une transition
        ...

    def decay_epsilon(self):
        # Decroitre le taux d'exploration
        ...

    def save(self, path):
        ...

    def load(self, path):
        ...
```

### Etape 2 : Enregistrer

Ajouter dans `algorithms/__init__.py` :

```python
from algorithms.mon_algo import MonAlgoAgent

registry.register_algorithm(
    "mon_algo",
    MonAlgoAgent,
    default_config={"lr": 0.1, "gamma": 0.99},
    compatible_envs=["taxi"],  # ou None pour tous les envs
)
```

C'est tout. L'algorithme apparait dans le menu.

---

## Ajouter un nouvel environnement

### Etape 1 : Creer le wrapper

Creer `environments/mon_env.py` avec une classe qui implemente `reset()`, `step(action)`, `close()` (API Gymnasium).

### Etape 2 : Enregistrer

Ajouter dans `environments/__init__.py` :

```python
from environments.mon_env import MonEnv

registry.register_environment(
    "mon_env",
    factory=lambda: MonEnv(),
    metadata={"n_states": 10, "n_actions": 4, "state_type": "discrete"},
)
```

---

## Ajouter un benchmark

Creer `benchmarks/mon_benchmark.py` en heritant de `BaseBenchmark`, puis enregistrer dans `benchmarks/__init__.py`.

---

## Algorithmes disponibles

| Algorithme   | Description                | Environnements compatibles |
| ------------ | -------------------------- | -------------------------- |
| `q_learning` | Q-Learning tabulaire       | Taxi-v3                    |
| `dqn`        | Double DQN (PyTorch, CUDA) | Taxi-v3, BeamNG            |

## Environnements disponibles

| Environnement | Description                                           | Type d'etat |
| ------------- | ----------------------------------------------------- | ----------- |
| `taxi`        | Gymnasium Taxi-v3 (500 etats discrets)                | Discret     |
| `beamng`      | BeamNG.drive conduite autonome (5 features continues) | Continu     |
