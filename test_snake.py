import random

import pytest

from players import GreedyPlayer
from simulation import play, simulate
from snake import DOWN, LEFT, RIGHT, UP, ShortSnake


@pytest.fixture(autouse=True)
def seed():
    random.seed(0)


# --- ShortSnake ---------------------------------------------------------------

def test_depart_au_milieu():
    game = ShortSnake(10)
    assert game.snake_pos == (5, 5)
    assert game.snake_dir == RIGHT
    assert game.score == 0
    assert game.fruit_pos != game.snake_pos


def test_none_continue_tout_droit():
    game = ShortSnake(10)
    game.fruit_pos = (0, 0)
    game.move(None)
    assert game.snake_pos == (5, 6)


def test_pas_de_demi_tour():
    game = ShortSnake(10)
    game.fruit_pos = (0, 0)
    game.move(LEFT)  # interdit : on continue vers la droite
    assert game.snake_pos == (5, 6)
    assert game.snake_dir == RIGHT
    assert LEFT not in game.actions()
    assert len(game.actions()) == 3


def test_bord_traverse_et_perd_10():
    game = ShortSnake(10)
    game.fruit_pos = (0, 0)
    game.restore_state((((5, 9),), RIGHT, (0, 0)))
    assert game.move(None) == -10
    assert game.snake_pos == (5, 0)
    assert game.score == -10


def test_fruit_rapporte_10_et_reapparait():
    game = ShortSnake(10)
    game.fruit_pos = (5, 6)
    assert game.move(None) == 10
    assert game.score == 10
    assert game.fruit_pos != game.snake_pos


def test_fruit_juste_apres_le_bord():
    game = ShortSnake(10)
    game.restore_state((((0, 3),), UP, (9, 3)))
    assert game.move(None) == 0  # -10 pour le bord, +10 pour le fruit


def test_save_restore():
    game = ShortSnake(10)
    saved = game.save_state()
    assert isinstance(saved, tuple)
    for d in [UP, UP, LEFT, DOWN, None]:
        game.move(d)
    assert game.save_state() != saved
    game.restore_state(saved)
    assert game.save_state() == saved
    assert game.snake_pos == saved[0][0]


def test_serpent_de_taille_2():
    game = ShortSnake(10, length=2)
    game.fruit_pos = (0, 0)
    assert game.body == [(5, 5), (5, 4)]
    game.move(UP)
    assert game.body == [(4, 5), (5, 5)]


def test_auto_intersection():
    game = ShortSnake(10, length=5)
    game.fruit_pos = (0, 0)
    assert game.body == [(5, 5), (5, 4), (5, 3), (5, 2), (5, 1)]
    # haut, gauche, bas : la tête retombe sur son propre corps
    assert game.move(UP) == 0
    assert game.move(LEFT) == 0
    assert game.move(DOWN) == -10


# --- GreedyPlayer (Q-Learning) ------------------------------------------------

def test_q_learning_mise_a_jour():
    game = ShortSnake(10)
    game.restore_state((((5, 9),), RIGHT, (0, 0)))
    player = GreedyPlayer(game, eps=0, gamma=0.5)
    s = game.save_state()
    s2 = (((5, 0),), RIGHT, (0, 0))
    player.action_values[(s2, UP)] = 4.0  # max_a' Q(s', a') = 4

    action = RIGHT
    player.last_state = s
    value = game.move(action)
    player.reward(action, value)
    # Q(s, a) <- 0 + 1/1 * (-10 + 0.5 * 4 - 0) = -8
    assert player.action_values[(s, action)] == pytest.approx(-8.0)
    assert player.eval_count[(s, action)] == 1


def test_greedy_prend_la_meilleure_action_de_l_etat():
    game = ShortSnake(10)
    player = GreedyPlayer(game, eps=0)
    s = game.save_state()
    player.action_values[(s, DOWN)] = 3.0
    player.action_values[(s, UP)] = 1.0
    assert {player._greedy_action() for _ in range(50)} == {DOWN}


def test_greedy_egalite_au_hasard():
    game = ShortSnake(10)
    player = GreedyPlayer(game, eps=0)
    assert {player._greedy_action() for _ in range(200)} == {UP, RIGHT, DOWN}


def test_actions_aleatoires_legales():
    game = ShortSnake(10)
    player = GreedyPlayer(game, eps=1)
    assert {player.get_action() for _ in range(200)} == {UP, RIGHT, DOWN}


def test_proportion_exploration():
    game = ShortSnake(10)
    player = GreedyPlayer(game, eps=0.3)
    s = game.save_state()
    player.action_values[(s, DOWN)] = 1.0
    n = 20_000
    other = sum(player.get_action() != DOWN for _ in range(n)) / n
    assert other == pytest.approx(0.3 * 2 / 3, abs=0.01)


def test_l_ia_apprend():
    rewards = simulate(runs=20, steps=10_000, size=5)
    debut = rewards[:, :1000].mean()
    fin = rewards[:, -1000:].mean()
    assert fin > debut + 1


def test_play_renvoie_le_score():
    game = ShortSnake(5)
    player = GreedyPlayer(game)
    rewards = play(game, player, 500)
    assert rewards.sum() == game.score
