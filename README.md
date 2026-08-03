# RL Pipeline

Pipeline de Reinforcement Learning pour **piloter vite** dans BeamNG.drive, puis
**courser** contre un autre agent ou contre un humain.

Une seule voiture (Cherrier Vivace Hillclimb, 682 ch AWD), un seul environnement
BeamNG, et deux axes de configuration independants :

| Axe | Valeurs | Choisi par |
|---|---|---|
| `sensor` (perception) | `lidar` (8 bins) / `adv_lidar` (grille 4x8) / `camera` (16x16) | l'utilisateur |
| `output` (actions) | `fixed` (table de 7 actions) / `continuous` (throttle, steering, brake) | **derive de l'algorithme** |

`output` n'est pas un champ du menu : une tete DQN ne peut pas emettre de commandes
continues et DDPG/TD3 n'emettent rien d'autre, donc l'algorithme le determine.

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
 ║ ◢◤ RL PIPELINE   Train › dqn › gridmap_v2         ⠹ running ║
 ╚════════════════════════════════════════════════════════════╝
 ┌ Workflows ───────┐┌ Train an agent ─────────────────────────┐
 │ ●  Train         ││ Algorithm    ‹ dqn ›                     │
 │ ○  Multi-agent   ││ Sensor       ‹ adv_lidar ›               │
 │ ○  Human play    ││ ── hyperparameters ──────────────────────│
 │ ○  Course (race) ││ lr 0.001   gamma 0.99   batch 64         │
 │ ○  Quit          ││          ▸ ▶ Start training              │
 │                  │└──────────────────────────────────────────┘
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

Le menu expose **quatre modes**.

### 1. Train an agent

- Choisir un algorithme (`dqn`, `dqn_per`, `ddpg`, `td3`) — il determine l'axe `output`
- Choisir le `sensor` (`lidar`, `adv_lidar`, `camera`) — il determine la largeur d'observation
- Ajuster les hyperparametres (ou garder les defaults)
- Le chemin de checkpoint est derive de `algo` + `sensor` + options, donc deux configs
  n'ecrasent jamais le meme fichier : `outputs/dqn_adv_lidar_h2_ori.pth`
- Pendant le run, la Status box affiche `checkpoints <n>` (checkpoints passes) en direct

### 2. Multi-agent training

- N voitures dans **une** scene, chacune sur **son propre** chemin (>= 30 m d'ecart),
  avec un seul pas physique pour tout le monde : c'est le mode "debit", sans contact
- Chaque voiture choisit son algorithme et son sensor

### 3. Human play

- Conduire manuellement, avec la lecture de l'observation en direct
- `lidar` / `adv_lidar` affichent les bins par cellule plus les diagnostics de filtrage ;
  `camera` affiche l'image du dashcam en ASCII, redessinee sur place

### 4. Course mode (race)

Deux voitures sur **le meme** trace, avec collisions.

- **Adversaire** : `algo` (deux checkpoints s'affrontent) ou `human` (vous conduisez)
- **Learning** : `false` = politiques gelees, bruit d'exploration a zero, donc la course
  montre ce que les checkpoints ont vraiment appris. `true` = les agents continuent
  d'apprendre avec le terme d'ecart, et les checkpoints sont sauves dans `outputs/races/`
- **Races** : nombre de courses consecutives ; le vainqueur et l'ecart en metres sont
  rapportes a chaque fois
- Un adversaire humain force le mode temps reel (personne ne peut conduire en lockstep) ;
  `algo` contre `algo` tourne en lockstep, donc deterministe et aussi rapide que le sim

> `laps` est reserve et doit valoir 1 : les traces generes sont des routes ouvertes, donc
> un second tour impliquerait de revenir au depart. Fermer un circuit reste a faire.

Les modes `evaluate`, `benchmark` et `generate trajectories` ont ete retires du menu ;
leur code reste importable (`core.pipeline_actions`, `benchmarks/`). Les caches de
trajectoires se generent desormais au premier lancement sur une map.

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
│   ├── runner.py            # Boucle train/eval generique (une voiture)
│   ├── multi_runner.py      # N agents, N chemins, un pas partage
│   ├── race_runner.py       # Course tete-a-tete (exhibition / apprentissage)
│   └── pipeline_actions.py  # Couche d'actions pilotee par l'UI
│
├── algorithms/
│   ├── __init__.py           # Registration des algorithmes
│   ├── dqn.py                # Double DQN + Dueling (+ PER)
│   ├── ddpg.py               # DDPG
│   └── td3.py                # TD3
│
├── environments/
│   ├── __init__.py           # Registration (un seul env : beamng)
│   ├── beamng_spec.py        # Les deux axes + toutes les tailles + la voiture
│   ├── beamng_sensors.py     # Construction / lecture des capteurs beamngpy
│   ├── beamng_geometry.py    # Math pure (lidar, progression, grille de depart)
│   ├── beamng_reward.py      # La recompense de course (partagee par les 3 envs)
│   ├── beamng.py             # L'env, parametre par sensor + output
│   ├── beamng_multi.py       # N voitures, chemins separes (entrainement)
│   └── beamng_race.py        # N voitures, meme trace, collisions (course)
│
├── benchmarks/               # Hors menu, toujours importable
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
    compatible_envs=["beamng"],
)
```

Puis le classer sur l'axe `output` dans `environments/beamng_spec.py` : ajouter son nom
a `FIXED_ALGOS` (tete discrete) ou `CONTINUOUS_ALGOS` (commandes continues).
`output_for_algo` leve une erreur sur un algorithme non classe plutot que de deviner —
un mauvais choix de tete d'action ne se verrait que bien plus tard, sous forme de
conduite absurde.

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

## Recompense de course

Une seule fonction, `environments/beamng_reward.compute_race_reward`, utilisee par les
trois environnements (solo, multi, course) — une politique est donc recompensee pour le
meme comportement partout.

| Terme | Role |
|---|---|
| progression vers le waypoint (`x3`) | signal dense, telescopique |
| vitesse projetee sur la direction cible (`x3`) | rouler vers la cible |
| **penalite par pas** | c'est ce qui fait qu'un tour *rapide* bat un tour propre mais lent |
| bonus de checkpoint **plat** + bonus de temps de segment | atteindre *chaque* checkpoint plus tot |
| bonus d'arrivee + budget de pas restant | le plus fort signal "va vite" |
| penalite de degats (adoucie), fin sur `MAX_DAMAGE` | le contact est tarife, pas fatal |
| penalite de proximite LiDAR, hors-piste gradue | rester sur la route |
| **terme d'ecart** (course seulement) | `GAP_COEF x (metres gagnes sur le rival)` |

Le terme d'ecart est telescopique : sa somme sur un episode vaut
`GAP_COEF x l'ecart final`, donc attaquer et defendre rapportent autant, et il est
impossible de le farmer en oscillant a cote du rival. Il ne s'active que si la
progression du rival est fournie ; en solo il ne contribue rien.

> Le bonus de checkpoint est volontairement **plat**. L'ancienne version payait
> `100 x waypoint_idx`, ce qui rendait la recompense de fin de trace un ordre de
> grandeur plus grande que celle du debut *pour une conduite identique* — de quoi
> destabiliser la fonction de valeur.

---

## Benchmarks

> Retire du menu (voir [Utilisation](#utilisation)) ; le code reste importable.

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
2. **Test** - Smoke tests (imports, registry, entrainement DQN sur un env jouet)

---

## Trajectoires automatiques (BeamNG)

Les waypoints, spawn position et spawn rotation sont desormais generes
automatiquement pour chaque map a partir du reseau routier (DecalRoads)
de BeamNG.

- Generes automatiquement au premier lancement sur une map (`load_or_generate`)
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

| Algorithme | Description                            | Axe `output` derive          |
| ---------- | -------------------------------------- | ---------------------------- |
| `dqn`      | Double DQN + Dueling (PyTorch, CUDA)   | `fixed` — table de 7 actions |
| `dqn_per`  | DQN avec Prioritized Experience Replay | `fixed` — table de 7 actions |
| `ddpg`     | DDPG                                   | `continuous` — 3 sorties     |
| `td3`      | TD3                                    | `continuous` — 3 sorties     |

## Environnement et axes

Un seul environnement enregistre, `beamng`, parametre par les deux axes. Les longueurs
d'observation sont inchangees par rapport aux anciennes classes par-capteur :

| `sensor`    | Bloc de perception              | `n_states` (sans option) |
| ----------- | ------------------------------- | ------------------------ |
| `lidar`     | 8 bins de distance (une rangee) | 14                       |
| `adv_lidar` | grille 4 x 8 (elevation x azimut) | 38                     |
| `camera`    | dashcam 16 x 16 en niveaux de gris | 262                   |

```
kinematic(6) | perception(P) | hints(2*H) | [pitch, roll]? | [edgeL, edgeR]?
```

Les tailles viennent toutes de `environments/beamng_spec.py` (`obs_size`,
`action_size`, `output_for_algo`) — une seule source de verite, au lieu des trois
copies de la meme arithmetique qui existaient avant.

`adv_lidar` echange de la resolution verticale contre un champ de vision plus etroit :
ses 4 rangees couvrent ainsi des elevations utiles au lieu de surtout du ciel et
surtout du bitume, ce qui permet a la politique de distinguer un mur (remplit toutes
les rangees) d'un obstacle bas (rangee du bas seulement).
