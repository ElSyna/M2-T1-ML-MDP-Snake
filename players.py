import random


class GreedyPlayer:
    """Joueur ε-greedy qui apprend Q(état, action) par Q-Learning."""

    def __init__(self, game, eps=0.1, gamma=0.5):
        self.game = game
        self.eps = eps
        self.gamma = gamma
        self.action_values = {}  # Q[(état, action)], 0 par défaut
        self.eval_count = {}     # N[(état, action)]
        self.last_state = None

    def get_action(self):
        self.last_state = self.game.save_state()
        explore = random.random() < self.eps
        if explore:
            return self._random_action()
        return self._greedy_action()

    def q(self, state, action):
        return self.action_values.get((state, action), 0.0)

    def _greedy_action(self):
        state = self.game.save_state()
        actions = self.game.actions()
        values = [self.q(state, a) for a in actions]
        best = max(values)
        return random.choice([a for a, v in zip(actions, values) if v == best])

    def _random_action(self):
        return random.choice(self.game.actions())

    def reward(self, action, value):
        # appelée après game.move(action) : le jeu est déjà dans l'état suivant
        key = (self.last_state, action)
        next_state = self.game.save_state()
        best_next = max(self.q(next_state, a) for a in self.game.actions())
        target = value + self.gamma * best_next

        self.eval_count[key] = self.eval_count.get(key, 0) + 1
        q = self.action_values.get(key, 0.0)
        self.action_values[key] = q + (target - q) / self.eval_count[key]
