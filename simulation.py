import numpy as np

from players import GreedyPlayer
from snake import ShortSnake


def play(game, player, steps):
    """Fait jouer `player` sur `game` pendant `steps` étapes.

    Renvoie la récompense obtenue à chaque étape.
    """
    rewards = np.zeros(steps)
    for t in range(steps):
        action = player.get_action()
        value = game.move(action)
        player.reward(action, value)
        rewards[t] = value
    return rewards


def simulate(runs, steps, size=10, length=1, eps=0.1, gamma=0.5):
    """Relance `runs` parties indépendantes, renvoie un tableau (runs, steps)."""
    all_rewards = np.zeros((runs, steps))
    for r in range(runs):
        game = ShortSnake(size, length=length)
        player = GreedyPlayer(game, eps=eps, gamma=gamma)
        all_rewards[r] = play(game, player, steps)
    return all_rewards
