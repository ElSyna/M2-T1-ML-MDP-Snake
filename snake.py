import random

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRECTIONS = [UP, RIGHT, DOWN, LEFT]
NAMES = {UP: "haut", RIGHT: "droite", DOWN: "bas", LEFT: "gauche"}


def opposite(direction):
    return (-direction[0], -direction[1])


class ShortSnake:
    def __init__(self, n, m=None, length=1):
        self.n = n
        self.m = n if m is None else m
        self.length = length
        self.grid = [["." for _ in range(self.m)] for _ in range(self.n)]

        # le serpent part du milieu vers la droite, la queue derrière la tête
        head = (self.n // 2, self.m // 2)
        self.snake_dir = RIGHT
        self.body = [(head[0], (head[1] - i) % self.m) for i in range(length)]
        self.snake_pos = head

        self.score = 0
        self.fruit_pos = None
        self._place_fruit()

    def _place_fruit(self):
        free = [(i, j) for i in range(self.n) for j in range(self.m)
                if (i, j) not in self.body]
        self.fruit_pos = random.choice(free)

    def actions(self):
        """Directions autorisées : tout sauf le demi-tour."""
        return [d for d in DIRECTIONS if d != opposite(self.snake_dir)]

    def move(self, direction=None):
        """Avance d'une case et renvoie la récompense (variation du score)."""
        if direction is not None and direction != opposite(self.snake_dir):
            self.snake_dir = direction

        reward = 0
        i = self.snake_pos[0] + self.snake_dir[0]
        j = self.snake_pos[1] + self.snake_dir[1]
        if not (0 <= i < self.n and 0 <= j < self.m):
            reward -= 10
            i, j = i % self.n, j % self.m

        self.snake_pos = (i, j)
        self.body = [self.snake_pos] + self.body[:-1]
        if self.length >= 5 and self.snake_pos in self.body[1:]:
            reward -= 10

        if self.snake_pos == self.fruit_pos:
            reward += 10
            self._place_fruit()

        self.score += reward
        return reward

    def save_state(self):
        return (tuple(self.body), self.snake_dir, self.fruit_pos)

    def restore_state(self, state):
        body, self.snake_dir, self.fruit_pos = state
        self.body = list(body)
        self.snake_pos = self.body[0]

    def __str__(self):
        rows = [row[:] for row in self.grid]
        for i, j in self.body[1:]:
            rows[i][j] = "o"
        fi, fj = self.fruit_pos
        rows[fi][fj] = "*"
        hi, hj = self.snake_pos
        rows[hi][hj] = "@"
        lines = [" ".join(row) for row in rows]
        lines.append(f"score : {self.score}")
        return "\n".join(lines)
