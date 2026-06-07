import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
import math

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
FLAP_POWER = -5
PIPE_SPEED = 3
PIPE_GAP = 150
PIPE_FREQUENCY = 1800  # milliseconds
GROUND_HEIGHT = 100


class Bird:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.radius = 20

    def flap(self):
        self.velocity = FLAP_POWER

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                          self.radius * 2, self.radius * 2)


class Pipe:
    def __init__(self, x=None):
        self.x = x if x is not None else SCREEN_WIDTH
        self.height = np.random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - PIPE_GAP - 50)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    def collide(self, bird):
        bird_rect = bird.get_rect()
        # Top pipe rect
        top_pipe_rect = pygame.Rect(self.x, 0, 70, self.height)
        # Bottom pipe rect
        bottom_pipe_rect = pygame.Rect(self.x, self.height + PIPE_GAP, 70,
                                      SCREEN_HEIGHT - self.height - PIPE_GAP - GROUND_HEIGHT)

        return bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect)

    def off_screen(self):
        return self.x < -70


class FlappyBirdEnv(gym.Env):
    """
    Flappy Bird environment for reinforcement learning.

    Observation space:
    - Bird's vertical position (normalized)
    - Bird's vertical velocity
    - Horizontal distance to the next pipe
    - Vertical distance to the top of the gap
    - Vertical distance to the bottom of the gap

    Action space:
    - 0: Do nothing
    - 1: Flap (jump)

    Reward function:
    - +1 for passing through a pipe
    - -1 for dying
    - 0 for each frame survived

    Episode termination:
    - When the bird collides with a pipe or the ground
    """

    metadata = {'render_modes': ['human'], 'render_fps': 60}

    def __init__(self, render_mode=None):
        super(FlappyBirdEnv, self).__init__()

        # Define action and observation space
        self.action_space = spaces.Discrete(2)  # 0: do nothing, 1: flap

        # Observation space:
        # [bird_y_normalized, bird_velocity, dist_to_pipe, top_pipe_gap, bottom_pipe_gap]
        self.observation_space = spaces.Box(
            low=np.array([0, -10, 0, -1, -1], dtype=np.float32),
            high=np.array([1, 10, 1, 1, 1], dtype=np.float32),
            dtype=np.float32
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize pygame only if rendering
        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Flappy Bird RL")
            self.clock = pygame.time.Clock()

        self.reset()

    def _get_obs(self):
        # Normalize bird's y position to [0, 1]
        bird_y_norm = self.bird.y / SCREEN_HEIGHT

        # Find the next pipe
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + 70 > self.bird.x:  # Pipe is ahead of bird
                next_pipe = pipe
                break

        if next_pipe is None:
            # If no pipe ahead, use default values
            dist_to_pipe = 1.0
            top_pipe_gap = 0.5
            bottom_pipe_gap = 0.5
        else:
            # Distance to pipe normalized by screen width
            dist_to_pipe = max(0, (next_pipe.x - self.bird.x) / SCREEN_WIDTH)

            # Gap positions normalized by screen height
            top_pipe_gap = next_pipe.height / SCREEN_HEIGHT
            bottom_pipe_gap = (next_pipe.height + PIPE_GAP) / SCREEN_HEIGHT

        # Normalize bird velocity
        bird_velocity = max(-10, min(10, self.bird.velocity))  # Clamp to [-10, 10]

        return np.array([
            bird_y_norm,
            bird_velocity,
            dist_to_pipe,
            top_pipe_gap,
            bottom_pipe_gap
        ], dtype=np.float32)

    def _get_info(self):
        return {
            "score": self.score,
            "birds_y_position": self.bird.y,
            "birds_velocity": self.bird.velocity
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset game state
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.frame_count = 0

        # Add initial pipe
        self.pipes.append(Pipe())

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Apply action
        if action == 1:  # Flap
            self.bird.flap()

        # Update bird
        self.bird.update()

        # Check if bird hit the ground or ceiling
        if (self.bird.y > SCREEN_HEIGHT - GROUND_HEIGHT - self.bird.radius or
            self.bird.y < self.bird.radius):
            self.game_over = True

        # Update pipes
        for pipe in self.pipes[:]:
            pipe.update()

            # Check if bird passed the pipe
            if not pipe.passed and pipe.x < self.bird.x:
                pipe.passed = True
                self.score += 1

            # Check for collision
            if pipe.collide(self.bird):
                self.game_over = True

            # Remove pipes that are off screen
            if pipe.off_screen():
                self.pipes.remove(pipe)

        # Add new pipes
        if len(self.pipes) == 0 or self.pipes[-1].x < SCREEN_WIDTH - 200:
            self.pipes.append(Pipe())

        # Get reward
        reward = 0
        if self.game_over:
            reward = -1  # Negative reward for dying
        elif any(pipe.passed and pipe.x + 70 < self.bird.x for pipe in self.pipes):
            reward = 1  # Positive reward for passing through a pipe

        # Get observation
        observation = self._get_obs()
        info = self._get_info()

        # Episode terminates when game is over
        terminated = self.game_over
        truncated = False  # We don't limit episode length

        if self.render_mode == "human":
            self._render_frame()

        self.frame_count += 1

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        if self.render_mode != "human":
            return

        # Fill the screen with blue (sky)
        self.screen.fill((0, 191, 255))  # Sky blue

        # Draw pipes
        for pipe in self.pipes:
            # Draw top pipe
            pygame.draw.rect(self.screen, (0, 128, 0), (pipe.x, 0, 70, pipe.height))
            # Draw bottom pipe
            bottom_pipe_y = pipe.height + PIPE_GAP
            bottom_pipe_height = SCREEN_HEIGHT - bottom_pipe_y - GROUND_HEIGHT
            pygame.draw.rect(self.screen, (0, 128, 0),
                           (pipe.x, bottom_pipe_y, 70, bottom_pipe_height))

        # Draw ground
        pygame.draw.rect(self.screen, (0, 128, 0),
                        (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))

        # Draw bird
        pygame.draw.circle(self.screen, (255, 255, 0), (int(self.bird.x), int(self.bird.y)), self.bird.radius)
        # Draw eye
        pygame.draw.circle(self.screen, (0, 0, 0), (int(self.bird.x + 10), int(self.bird.y - 5)), 5)

        # Draw score
        font = pygame.font.Font(None, 36)
        score_text = font.render(str(self.score), True, (255, 255, 255))
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 50))

        # Draw game over message
        if self.game_over:
            game_over_font = pygame.font.Font(None, 48)
            game_over_text = game_over_font.render("Game Over", True, (255, 0, 0))
            restart_text = font.render("Reset to restart", True, (255, 255, 255))
            self.screen.blit(game_over_text,
                           (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(restart_text,
                           (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2 + 10))

        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.render_mode == "human":
            pygame.quit()