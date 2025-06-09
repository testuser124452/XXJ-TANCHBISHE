import random

import pygame

# 定义常量
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 1000
POP_SIZE = 1000     # 大幅提高训练轮数
BLOCK_SIZE = 25

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

class Snake:
    def __init__(self):
        self.length = 3
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
        self.color = GREEN

    def get_head_position(self):
        return self.positions[0]

    def turn(self, point):
        if (point[0] * -1, point[1] * -1) == self.direction:
            return
        else:
            self.direction = point

    def move(self):
        cur = self.get_head_position()
        x, y = self.direction
        new = ((cur[0] + (x * BLOCK_SIZE)) % SCREEN_WIDTH, (cur[1] + (y * BLOCK_SIZE)) % SCREEN_HEIGHT)
        self.positions.insert(0, new)
        if len(self.positions) > self.length:
            self.positions.pop()

    def reset(self):
        self.length = 3
        self.positions = [(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)]
        self.direction = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, self.color, r)
            pygame.draw.rect(surface, BLACK, r, 1)

class Food:
    def __init__(self, special=False):
        x = random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE)
        y = random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE)
        self.position = (x, y)
        if special:
            self.type = random.choice(['bonus', 'poison', 'slow', 'fast'])
            if self.type == 'bonus':
                self.color = (0, 0, 255)
            elif self.type == 'poison':
                self.color = (255, 0, 255)
            elif self.type == 'slow':
                self.color = (255, 255, 0)
            elif self.type == 'fast':
                self.color = (0, 255, 255)
        else:
            self.type = 'normal'
            self.color = RED

    def get_position(self):
        return self.position

    def draw(self, surface):
        r = pygame.Rect((self.position[0], self.position[1]), (BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(surface, self.color, r)
        pygame.draw.rect(surface, BLACK, r, 1)

class Obstacle:
    def __init__(self, num=20):
        self.positions = []
        for _ in range(num):
            x = random.randrange(0, SCREEN_WIDTH, BLOCK_SIZE)
            y = random.randrange(0, SCREEN_HEIGHT, BLOCK_SIZE)
            self.positions.append((x, y))

    def draw(self, surface):
        for p in self.positions:
            r = pygame.Rect((p[0], p[1]), (BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(surface, (80, 80, 80), r)
