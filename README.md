# RL Pipeline

Pipeline de Reinforcement Learning modulaire pour entrainer et evaluer des agents sur differents environnements (Taxi-v3, BeamNG.drive).

Ajouter un nouvel algorithme = **1 fichier + 1 ligne de registration**. Il apparait automatiquement dans le menu.

---

## Requirements

- Python 3.11
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

`python main.py` lance l'application OpenTUI via Bun (installer Bun au prealable : https://bun.sh).
L'interface terminal moderne (« command center ») se navigue au clavier :

```
 ╔════════════════════════════════════════════════════════════╗
 ║ ◢◤ RL PIPELINE   Train › dqn › beamng_lidar       ⠹ running ║
 ╚════════════════════════════════════════════════════════════╝
 ┌ Workflows ───────┐┌ Train an agent ─────────────────────────┐
 │ ●  Train         ││ Algorithm    ‹ dqn ›                     │
 │ ○  Evaluate      ││ Environment  ‹ beamng_lidar ›            │
 │ ○  Benchmark     ││ ── hyperparameters ──────────────────────│
 │ ○  Human play    ││ lr 0.001   gamma 0.99   batch 64         │
 │ ○  Trajectories  ││          ▸ ▶ Start training              │
 │ ○  Multi-agent   │└──────────────────────────────────────────┘
 ├ Status ──────────┤┌ Output · l for full logs ──────────────┐
 │ ⠹ Train ▓▓▓░ 42% ││ ep 210/500  reward 6.4  ε 0.18          │
 └──────────────────┘└──────────────────────────────────────────┘
  ? help · l logs · ⇥ field · ⏎ run · ← → choice · esc back · ^C quit
```

Touches : `↑ ↓` naviguer dans le menu, `Enter` ouvrir un workflow / lancer le bouton
focalisé, `Tab` champ suivant, `← →` changer un choix, `l` ouvrir la fenêtre des logs
complets, `?` aide clavier, `Esc` revenir au menu / fermer une fenêtre, `Ctrl+C` annuler
un run en cours ou quitter. Le panneau Output est un aperçu compact ; `l` ouvre les logs
complets (défilables). L'UI intègre une barre de progression live (lue depuis la sortie
d'entrainement), la validation des champs numeriques et un overlay d'aide clavier.

### 1. Train an agent

- Choisir un algorithme (`q_learning`, `dqn`, ...)
- Choisir un environnement compatible
- Ajuster les hyperparametres (ou garder les defaults)
- L'entrainement demarre, les modeles et plots sont sauvegardes dans `outputs/`

### 2. Evaluate an agent

- Charger un modele sauvegarde
- Lancer N episodes en mode exploitation (epsilon=0)

### 3. Run a benchmark

- Choisir un benchmark (`convergence`, `comparison`, `gridsearch`)
- Saisir les **seeds** (defaut `0,1,2,3,4`), le nombre d'**episodes d'evaluation** et le **seuil de succes**
- Les resultats sont affiches a la fin et exportes dans `outputs/benchmarks/`

Voir la section [Benchmarks](#benchmarks) pour le detail des metriques et des fichiers produits.

### 4. Human play (BeamNG)

- Conduire manuellement dans BeamNG pour tester le scenario

---

## Structure du projet

```
testRomain/
├── main.py                  # Point d'entree -> launcher OpenTUI
├── config.py                # Charge les variables depuis .env
├── .env                     # Variables d'environnement (non versionne)
├── .env.template            # Template a copier
│
├── core/
│   ├── base_agent.py        # Classe abstraite BaseAgent
│   ├── base_benchmark.py    # Classe abstraite BaseBenchmark
│   ├── registry.py          # Registre central (algos, envs, benchmarks)
│   ├── runner.py            # Boucle train/eval generique
│   └── pipeline_actions.py  # Couche d'actions pilotee par l'UI
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
├── tui/                     # Application terminal OpenTUI (Bun + TypeScript)
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

## Benchmarks

La suite de benchmarks est **agnostique** : chaque benchmark tourne sur
n'importe quel couple algorithme + environnement enregistre. Trois principes
la rendent fiable et reproductible :

- **Seeds** : chaque run est seede (RNG globaux + environnement). Une meme seed
  redonne exactement le meme resultat.
- **Multi-seed** : chaque configuration est rejouee sur plusieurs seeds et les
  metriques sont agregees en `mean ± std`, intervalle de confiance a 95 % (CI95),
  min et max. Un seul run ne prouve rien ; la moyenne inter-seeds, si.
- **Evaluation gloutonne** : apres l'entrainement, la politique est evaluee a
  `epsilon=0` sur N episodes pour mesurer sa **vraie** performance (taux de
  succes, recompense, pas/episode), pas seulement la recompense d'entrainement.

### Benchmarks disponibles

| Benchmark     | Role                                                                 |
| ------------- | -------------------------------------------------------------------- |
| `convergence` | Analyse complete d'un algo : vitesse de convergence, stabilite, distribution |
| `comparison`  | Compare plusieurs algos sur un meme environnement (multi-seed)       |
| `gridsearch`  | Balaye une grille d'hyperparametres et classe les configurations     |

### Metriques rapportees

| Categorie            | Metriques                                                        |
| -------------------- | ---------------------------------------------------------------- |
| Vitesse              | `convergence_episode`, `improvement_rate`                        |
| Perf entrainement    | `final_avg_reward` (20 % derniers ep.), `best/worst`             |
| Perf reelle (eval)   | `eval_mean_reward`, `eval_std_reward`, `eval_success_rate`, `eval_mean_steps` |
| Robustesse           | ecart-type inter-seeds, `ci95`                                   |
| Cout                 | `training_time_s`                                                |
| Reproductibilite     | seeds, commit git, versions (python/numpy/torch), device         |

### Fichiers de sortie

Chaque execution cree un dossier horodate dans `outputs/benchmarks/` :

```
outputs/benchmarks/<benchmark>_<algo>_<env>_<timestamp>/
├── report.md           # rapport lisible (tables + images + reproductibilite)
├── metadata.json       # commit git, seeds, versions, device
├── summary.json        # metriques agregees (multi-seed)
├── results_full.json   # tout, courbes brutes par seed incluses
├── metrics.csv         # 1 ligne par seed (ou par run) — exploitable Excel/pandas
├── summary.csv         # 1 ligne par metrique/variante agregee
└── *.png               # courbes (mean ± std), barres, heatmap (gridsearch)
```

`gridsearch` produit en plus un `leaderboard.csv` (configurations classees) et
une `heatmap.png` lorsque deux hyperparametres sont balayes.

## Linting (Ruff)

Le projet utilise [Ruff](https://docs.astral.sh/ruff/) pour le linting et le formatage. La config est dans `ruff.toml`.

Verifier le code :

```bash
ruff check .
```

Corriger automatiquement les erreurs :

```bash
ruff check --fix .
```

Formater le code :

```bash
ruff format .
```

---

## CI

Le projet utilise GitHub Actions pour l'integration continue (`.github/workflows/ci.yml`).

Le pipeline se declenche sur chaque push/PR vers `main` ou `romain_test` et execute :

1. **Lint & Format** - `ruff check .` + `ruff format --check .`
2. **Test** - Smoke tests (imports, registry, entrainement Q-Learning sur Taxi)

---

## Trajectoires automatiques (BeamNG)

Les waypoints, spawn position et spawn rotation sont desormais generes
automatiquement pour chaque map a partir du reseau routier (DecalRoads)
de BeamNG.

- Pre-calcul depuis l'app OpenTUI : `Generate trajectories (BeamNG)`
- Cache sur disque : `outputs/trajectories/<map>.json`
- Pour regenerer : supprimer le fichier JSON ou relancer l'option avec `Overwrite`
- Pour les maps sans routes (`smallgrid`), une boucle carree de 80 m sert de fallback

Le format du cache JSON :

```json
{
  "spawn_pos": [x, y, z],
  "spawn_rot": [qx, qy, qz, qw],
  "sparse_waypoints": [[x, y, z], ...],
  "dense_waypoints":  [[x, y, z], ...],
  "map_name": "...",
  "generated_at": "...",
  "source": "road_network:<id>" | "fallback:square_loop"
}
```

Si vous voulez surcharger la trajectoire pour une map particuliere, editez
le JSON a la main ou utilisez la procedure decrite dans `scenario_creator.md`.

---

## Algorithmes disponibles

| Algorithme   | Description                              | Environnements compatibles |
| ------------ | ---------------------------------------- | -------------------------- |
| `q_learning` | Q-Learning tabulaire                     | Taxi-v3                    |
| `dqn`        | Double DQN + Dueling (PyTorch, CUDA)     | Taxi-v3, BeamNG            |
| `dqn_per`    | DQN avec Prioritized Experience Replay   | Taxi-v3, BeamNG            |
| `ddpg`       | DDPG (actions continues)                 | BeamNG                     |
| `td3`        | TD3 (actions continues)                  | BeamNG                     |

## Environnements disponibles

| Environnement | Description                                           | Type d'etat |
| ------------- | ----------------------------------------------------- | ----------- |
| `taxi`        | Gymnasium Taxi-v3 (500 etats discrets)                | Discret     |
| `beamng`      | BeamNG.drive conduite autonome (5 features continues) | Continu     |
