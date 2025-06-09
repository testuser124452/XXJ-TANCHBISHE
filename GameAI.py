import pygame
import numpy as np
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import trange
from sklearn.metrics import roc_curve, auc

from Game import Snake, Food, Obstacle, SCREEN_WIDTH, SCREEN_HEIGHT, BLOCK_SIZE, POP_SIZE, WHITE
from network import SnakeAI
from logger_utils import log

pygame.init()
font = pygame.font.SysFont('comics', 30)


def get_direction(action):
    if action == 0:
        return 0, -1
    elif action == 1:
        return 0, 1
    elif action == 2:
        return -1, 0
    else:
        return 1, 0


def load_best_score(filepath='best_score.txt'):
    try:
        with open(filepath, 'r') as f:
            score = int(f.read())
        print(f"Loaded best score from {filepath}: {score}")
        return score
    except Exception:
        return 0


class Game:
    def save_best_score(self, filepath='best_score.txt'):
        with open(filepath, 'w') as f:
            f.write(str(self.best_score))

    def __init__(self, buffer_size=15000, batch_size=64):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.snake = Snake()
        self.food = Food()
        self.obstacle = Obstacle(num=15)
        self.ai_player = SnakeAI(buffer_size, batch_size)
        self.scores = []
        self.best_score = load_best_score()
        self.i = 0

        self.epoch_scores = []
        self.epoch_steps = []
        self.epoch_rewards = []
        self.epoch_loss = []
        self.epoch_pred = []
        self.epoch_labels = []

        self.ai_player.load_best_weights()

    def generate_food(self, special=False):
        while True:
            food = Food(special)
            if (food.position not in self.snake.positions) and (food.position not in self.obstacle.positions):
                return food

    def update(self, ai_player, tran_i):
        state = self.get_state()
        action = ai_player.get_action(state)
        v_a = self.snake.direction
        v_b = get_direction(action)
        self.snake.turn(get_direction(action))

        if v_a != v_b and (v_a[0] * -1, v_a[1] * -1) != v_b:
            self.i = 0

        distances = np.sqrt(np.sum((np.array(self.snake.get_head_position()) - np.array(self.food.position)) ** 2))
        self.snake.move()
        done = False
        reward = 0

        max_len = (SCREEN_WIDTH // BLOCK_SIZE) * (SCREEN_HEIGHT // BLOCK_SIZE) - len(self.obstacle.positions)

        # 吃到食物判定与效果
        if self.snake.get_head_position() == self.food.position:
            if self.food.type == 'bonus':
                self.snake.length += 3
                reward += 20
            elif self.food.type == 'poison':
                self.snake.length = max(3, self.snake.length - 2)
                reward -= 6
            elif self.food.type == 'slow':
                self.clock.tick(18)
            elif self.food.type == 'fast':
                self.clock.tick(30)
            else:
                self.snake.length += 1
                reward += 20
            self.food = self.generate_food(special=(np.random.rand() < 0.2))

            # 吃满全屏检测与终极大奖励（你的建议）
            if self.snake.length == max_len:
                reward += 500
                done = True

        if self.is_collision():
            self.scores.append(self.snake.length)
            done = True

        next_state = self.get_state()
        reward += self.get_reward(done, distances, tran_i)
        ai_player.add_experience(state, action, reward, next_state, done)
        ai_player.train_model()

        # ROC统计
        state_reshape = np.reshape(state, [1, 12])
        probs = ai_player.model.predict(state_reshape, verbose=0)[0]
        self.epoch_pred.append(np.max(probs))
        self.epoch_labels.append(int(done))

        return done

    def is_collision(self):
        head = self.snake.get_head_position()
        return (head in self.snake.positions[1:]) or (head in self.obstacle.positions)

    def get_accessible_space(self):
        from collections import deque
        visited = set()
        queue = deque()
        head = self.snake.get_head_position()
        queue.append(head)
        visited.add(head)
        directions = [(-BLOCK_SIZE, 0), (BLOCK_SIZE, 0), (0, -BLOCK_SIZE), (0, BLOCK_SIZE)]
        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                np_pos = (nx, ny)
                if (0 <= nx < SCREEN_WIDTH and 0 <= ny < SCREEN_HEIGHT
                        and np_pos not in self.snake.positions[1:]
                        and np_pos not in self.obstacle.positions
                        and np_pos not in visited):
                    visited.add(np_pos)
                    queue.append(np_pos)
        return len(visited)

    def get_reward(self, done, distances, tran_i):
        # ——奖励函数：你建议的最终版——
        distances_2 = np.sqrt(np.sum((np.array(self.snake.get_head_position()) - np.array(self.food.position)) ** 2))
        reward = 0

        # 步数惩罚
        if tran_i > 20:
            reward -= 0.15
        # 死亡
        if done:
            reward -= 50
        # 吃到食物
        elif self.snake.get_head_position() == self.food.position:
            reward += 20
        # 距离食物变近
        elif distances_2 < distances:
            reward += 1
        else:
            reward -= 0.3

        # ——空间奖励（核心！）——
        accessible_space = self.get_accessible_space()
        total_space = (SCREEN_WIDTH // BLOCK_SIZE) * (SCREEN_HEIGHT // BLOCK_SIZE)
        ratio = accessible_space / total_space
        if ratio < 0.08:
            reward -= 20  # 极度危险
        elif ratio < 0.13:
            reward -= 12
        elif ratio < 0.18:
            reward -= 6
        elif ratio < 0.24:
            reward -= 2
        elif ratio > 0.7:
            reward += 4
        elif ratio > 0.5:
            reward += 2
        elif ratio > 0.38:
            reward += 1
        elif ratio > 0.26:
            reward += 0.5

        # ——鼓励靠近蛇尾（高阶技巧，防绕死）——
        tail = self.snake.positions[-1]
        dist_tail = np.linalg.norm(np.array(self.snake.get_head_position()) - np.array(tail))
        reward += 0.01 * dist_tail / BLOCK_SIZE

        return reward

    def get_state(self):
        head = self.snake.get_head_position()
        food = self.food.position

        left = (head[0] - BLOCK_SIZE, head[1])
        right = (head[0] + BLOCK_SIZE, head[1])
        up = (head[0], head[1] - BLOCK_SIZE)
        down = (head[0], head[1] + BLOCK_SIZE)

        state = [
            (left in self.snake.positions[1:] or left in self.obstacle.positions),
            (right in self.snake.positions[1:] or right in self.obstacle.positions),
            (up in self.snake.positions[1:] or up in self.obstacle.positions),
            (down in self.snake.positions[1:] or down in self.obstacle.positions),
            food[0] < head[0],
            food[0] > head[0],
            food[1] < head[1],
            food[1] > head[1],
            self.snake.direction == (0, -1),
            self.snake.direction == (0, 1),
            self.snake.direction == (-1, 0),
            self.snake.direction == (1, 0),
        ]
        return np.asarray(state, dtype=np.float32)

    def run(self):
        log("Training Start", "INFO")
        try:
            for epoch in trange(POP_SIZE, desc="Epoch Progress", dynamic_ncols=True):
                self.snake.reset()
                self.obstacle = Obstacle(num=15)
                self.food = self.generate_food(special=(np.random.rand() < 0.2))
                done = False
                score = 0
                self.i = 0
                total_reward = 0
                losses = []
                while not done:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                            quit()
                    self.i += 1
                    done = self.update(ai_player=self.ai_player, tran_i=self.i)
                    score = self.snake.length
                    total_reward += self.ai_player.buffer[-1][2]
                    losses.append(self.ai_player.last_loss)
                    self.screen.fill(WHITE)
                    self.snake.draw(self.screen)
                    self.food.draw(self.screen)
                    self.obstacle.draw(self.screen)
                    score_text = font.render(f"Score: {score}  Best: {self.best_score}", True, (0, 0, 0))
                    self.screen.blit(score_text, (10, 10))
                    epoch_text = font.render(f"Epoch: {epoch + 1}/{POP_SIZE}", True, (0, 0, 0))
                    self.screen.blit(epoch_text, (10, 40))
                    pygame.display.update()
                    self.clock.tick(25)
                self.epoch_scores.append(score)
                self.epoch_steps.append(self.i)
                self.epoch_rewards.append(total_reward)
                self.epoch_loss.append(np.mean(losses) if losses else 0)

                if score > self.best_score:
                    self.best_score = score
                    self.ai_player.save_best_weights('best_weights.weights.h5')
                    self.save_best_score('best_score.txt')
                    log(f"New best score: {self.best_score}, weights saved!", "INFO")

                if (epoch + 1) % 10 == 0:
                    avg_score = np.mean(self.epoch_scores[-10:])
                    avg_steps = np.mean(self.epoch_steps[-10:])
                    avg_reward = np.mean(self.epoch_rewards[-10:])
                    avg_loss = np.mean(self.epoch_loss[-10:])
                    log(f"Epoch {epoch + 1}: Score={score}, Best={self.best_score}, AvgScore(10)={avg_score:.2f}, "
                        f"AvgSteps(10)={avg_steps:.2f}, AvgReward(10)={avg_reward:.2f}, AvgLoss(10)={avg_loss:.4f}",
                        "INFO")

            df = pd.DataFrame({
                'score': self.epoch_scores,
                'steps': self.epoch_steps,
                'reward': self.epoch_rewards,
                'loss': self.epoch_loss
            })


        except Exception as e:
            log(f"Exception occurred: {str(e)}", "ERROR")
        finally:
            log("Training finished.", "INFO")


if __name__ == "__main__":
    game = Game(buffer_size=15000, batch_size=64)
    game.run()
