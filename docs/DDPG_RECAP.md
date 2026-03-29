# DDPG - Notes

## L'algo DDPG (`algorithms/ddpg.py`)

DDPG est un algo actor-critic qui sort des **actions continues** (pas des choix discrets comme DQN).

- L'Actor decide quoi faire : il sort 2 valeurs `[acceleration, steering]` entre -1 et 1
- Le Critic evalue si c'etait un bon choix en estimant la Q-value
- Les 2 ont des target networks qui se mettent a jour doucement (soft update) pour stabiliser l'apprentissage
- Pour l'exploration on utilise du bruit Ornstein-Uhlenbeck qui diminue au fil du temps (epsilon decay)

### Pourquoi continu et pas discret ?

Au debut on avait fait DDPG avec 7 actions discretes (comme DQN) + argmax. Ca marchait pas car DDPG a besoin de faire passer le gradient a travers les actions. Avec argmax le gradient est coupe, l'actor n'apprend rien. En passant a 2 sorties continues le gradient remonte correctement du Critic vers l'Actor.

---

## Ce qu'on a change dans l'environnement (`environments/beamng.py`)

### 15 waypoints au lieu de 3

Avec 3 waypoints espaces de ~100m, l'agent perdait le signal de reward entre les checkpoints. Il savait pas ou aller et tournait en rond. En mettant 15 waypoints espaces de ~15-20m, il a toujours un objectif proche et recoit du feedback a chaque step.

### Reward specifique DDPG

Le reward par defaut (pour DQN) marchait pas bien avec DDPG. On a fait un reward separe `_compute_reward_ddpg()` avec :

- **Progress reward** : quand la voiture se rapproche du waypoint elle gagne des points, quand elle s'eloigne elle en perd. C'est le signal principal qui dit "va par la"
- **Vitesse projetee** : on recompense `speed * cos(heading_error)`. Ca veut dire que seule la vitesse vers le waypoint compte. Rouler vite dans la mauvaise direction donne un reward negatif. Tourner en rond donne ~0
- **Bonus d'alignement** : petit bonus quand la voiture pointe vers le waypoint, meme a l'arret. Ca l'aide a se reorienter
- **Penalite LiDAR** : si un obstacle est a moins de 10m penalite forte, entre 10 et 20m penalite legere. Sans ca l'agent foncait dans les murs car il ne recevait un signal negatif qu'apres le crash (trop tard)

### Pas de brake

On a remappe la sortie acceleration de [-1,1] vers throttle [0,1]. La voiture avance toujours. Sinon au debut de l'entrainement les poids sont aleatoires et ~50% du temps l'agent freinait, il n'apprenait rien.

### Mode multi-algo

L'env s'adapte selon l'algo :

- DQN/Q-learning : 3 waypoints, 7 actions discretes, reward original
- DDPG : 15 waypoints, 2 actions continues, reward DDPG

C'est selectionne automatiquement dans le CLI via `reward_mode`.

---

## Hyperparametres et pourquoi

- **`warmup_steps: 128`** : au debut c'etait 500, l'agent crashait 5 fois avant de commencer a apprendre. A 128 il commence apres 1-2 episodes
- **`epsilon_decay: 0.99`** : c'etait 0.998, apres 100 episodes l'agent explorait encore a 82%. A 0.99 il exploite plus vite ce qu'il a appris
- **`updates_per_step: 4`** : au lieu de 1 seul gradient update par step, on en fait 4. L'agent apprend 4x plus vite de chaque experience
- **Pas de reward normalization** : on avait une normalisation (mean=0, std=1) sur les rewards dans le batch. Ca effacait la difference entre un crash (-30) et une bonne action (+5), tout devenait ~0. En la supprimant le signal reste clair

---

## Resume training

Le checkpoint sauvegarde le numero d'episode + epsilon. Quand on relance, le CLI demande :

- **Continue** : reprend ou on en etait (memes poids, meme epsilon, bon numero d'episode)
- **Reset** : supprime le checkpoint et repart de zero

---

## Reward total

**Avant (3 waypoints)** : max ~800 de checkpoints, pas de reward entre les waypoints

**Maintenant (15 waypoints)** : ~950 de checkpoints + ~800-2000 de reward continu (progress + vitesse) selon la qualite de la conduite
