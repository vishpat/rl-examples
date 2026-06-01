import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register, registry
import numpy as np
import pygame
import random
from enum import Enum
from collections import namedtuple

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

class SnakeEnv(gym.Env):
    """
    A custom Gymnasium environment for the classic Snake game, based on the implementation
    in `snake_game.py`. This environment is fully compatible with stable-baselines3.
    
    Metadata:
        render_modes: "human" (interactive GUI) and "rgb_array" (returns np.ndarray of pixel data)
        render_fps: speed of the rendering in human mode
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 15}

    def __init__(self, w=640, h=480, render_mode=None, obs_type="features", reward_shaping=True):
        super(SnakeEnv, self).__init__()
        
        self.w = w
        self.h = h
        self.render_mode = render_mode
        self.obs_type = obs_type
        self.reward_shaping = reward_shaping
        self.block_size = 20
        
        # Action space: 4 discrete directions in clockwise order:
        # 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        self.action_space = spaces.Discrete(4)
        
        # Define observation space based on obs_type
        if self.obs_type == "features":
            # 11-element feature vector:
            # - Danger straight, danger right, danger left (3 elements, relative to current direction)
            # - Direction one-hot: up, right, down, left (4 elements)
            # - Food location relative to head: up, right, down, left (4 elements)
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(11,), dtype=np.float32)
        elif self.obs_type == "grid":
            # 3D grid representation: (3, grid_height, grid_width)
            # Channel 0: Snake Head, Channel 1: Snake Body, Channel 2: Food
            # Fully normalized binary masks suitable for CnnPolicy.
            self.grid_w = self.w // self.block_size
            self.grid_h = self.h // self.block_size
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, self.grid_h, self.grid_w), dtype=np.float32)
        elif self.obs_type == "rgb":
            # RGB screen pixels: (height, width, 3)
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.h, self.w, 3), dtype=np.uint8)
        else:
            raise ValueError(f"Unsupported obs_type: '{obs_type}'. Choose from 'features', 'grid', or 'rgb'.")
            
        # Pygame assets and state
        self.display = None
        self.clock = None
        self.font = None
        
        # RGB Colors
        self.WHITE = (255, 255, 255)
        self.RED = (200, 0, 0)
        self.BLUE1 = (0, 0, 255)
        self.BLUE2 = (0, 100, 255)
        self.BLACK = (0, 0, 0)
        
        # Track steps since last food to prevent infinite loops
        self.steps_no_food = 0
        self.total_steps = 0
        
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        Returns the initial observation and info dictionary.
        """
        super().reset(seed=seed)
        
        # Seed generator
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Reset game states
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - self.block_size, self.head.y),
            Point(self.head.x - (2 * self.block_size), self.head.y)
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        
        self.steps_no_food = 0
        self.total_steps = 0
        
        # Initialize rendering if needed
        if self.render_mode is not None and self.display is None:
            self._init_render()
            
        obs = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
            
        return obs, info

    def step(self, action):
        """
        Executes one step in the environment.
        Takes action and returns: (observation, reward, terminated, truncated, info)
        """
        # Convert action to scalar integer if it's a numpy type
        if isinstance(action, (np.ndarray, np.integer)):
            action = int(action.item())
            
        self.total_steps += 1
        self.steps_no_food += 1
        
        # Directions mapping: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
        idx_to_dir = {
            0: Direction.UP,
            1: Direction.RIGHT,
            2: Direction.DOWN,
            3: Direction.LEFT
        }
        dir_to_idx = {
            Direction.UP: 0,
            Direction.RIGHT: 1,
            Direction.DOWN: 2,
            Direction.LEFT: 3
        }
        
        new_direction = idx_to_dir[action]
        current_idx = dir_to_idx[self.direction]
        
        # Ignore direction change if it is directly opposite to current heading
        if (action - current_idx) % 2 != 0:
            self.direction = new_direction
            
        # Track distance to food before moving
        old_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        # Move snake head
        self._move(self.direction)
        self.snake.insert(0, self.head)
        
        # Track distance to food after moving
        new_dist = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        
        terminated = False
        reward = 0.0
        
        # Check collision
        if self._is_collision():
            terminated = True
            reward = -10.0
            
            obs = self._get_obs()
            info = self._get_info()
            if self.render_mode == "human":
                self._render_frame()
            return obs, reward, terminated, False, info
            
        # Check if food eaten
        if self.head == self.food:
            self.score += 1
            reward = 10.0
            self.steps_no_food = 0
            self._place_food()
        else:
            self.snake.pop()
            # Small penalty to encourage efficiency and discourage circular wandering
            reward = -0.01
            
            # Apply distance-based reward shaping
            if self.reward_shaping:
                if new_dist < old_dist:
                    reward += 0.11  # Reward for moving closer to food
                else:
                    reward -= 0.12  # Penalty for moving away from food
            
        # Check if the episode is truncated due to step loop protection
        # Scale step limit with the grid area to ensure the agent has enough exploration steps
        truncated = False
        grid_area = (self.w // self.block_size) * (self.h // self.block_size)
        max_steps = max(grid_area * 3, 200, len(self.snake) * 15)
        if self.steps_no_food >= max_steps:
            truncated = True
            
        # Handle GUI events in human rendering mode
        if self.render_mode == "human":
            self._render_frame()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    terminated = True
                    
        obs = self._get_obs()
        info = self._get_info()
        
        return obs, reward, terminated, truncated, info

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += self.block_size
        elif direction == Direction.LEFT:
            x -= self.block_size
        elif direction == Direction.DOWN:
            y += self.block_size
        elif direction == Direction.UP:
            y -= self.block_size
            
        self.head = Point(x, y)

    def _is_collision(self, pt=None):
        if pt is None:
            pt = self.head
            
        # hits boundary
        if pt.x > self.w - self.block_size or pt.x < 0 or pt.y > self.h - self.block_size or pt.y < 0:
            return True
        # hits itself (checking if pt is in body. If pt is head, we exclude index 0)
        if pt == self.head:
            return pt in self.snake[1:]
        else:
            return pt in self.snake
            
    def _place_food(self):
        grid_w = self.w // self.block_size
        grid_h = self.h // self.block_size
        
        # Use set subtraction to find all empty slots on the board
        all_points = {Point(x * self.block_size, y * self.block_size) for x in range(grid_w) for y in range(grid_h)}
        empty_points = list(all_points - set(self.snake))
        
        if empty_points:
            self.food = random.choice(empty_points)
        else:
            # Snake has won/filled the screen entirely
            self.food = self.head

    def _get_obs(self):
        if self.obs_type == "features":
            # Current action index mapping
            dir_to_idx = {
                Direction.UP: 0,
                Direction.RIGHT: 1,
                Direction.DOWN: 2,
                Direction.LEFT: 3
            }
            dir_idx = dir_to_idx[self.direction]
            
            # Predict collisions for relative movements
            pt_straight = self._get_next_point(dir_idx)
            pt_right = self._get_next_point((dir_idx + 1) % 4)
            pt_left = self._get_next_point((dir_idx - 1) % 4)
            
            danger_straight = 1.0 if self._is_collision(pt_straight) else 0.0
            danger_right = 1.0 if self._is_collision(pt_right) else 0.0
            danger_left = 1.0 if self._is_collision(pt_left) else 0.0
            
            # Current heading direction
            dir_up = 1.0 if self.direction == Direction.UP else 0.0
            dir_right = 1.0 if self.direction == Direction.RIGHT else 0.0
            dir_down = 1.0 if self.direction == Direction.DOWN else 0.0
            dir_left = 1.0 if self.direction == Direction.LEFT else 0.0
            
            # Food relative positions
            food_up = 1.0 if self.food.y < self.head.y else 0.0
            food_down = 1.0 if self.food.y > self.head.y else 0.0
            food_left = 1.0 if self.food.x < self.head.x else 0.0
            food_right = 1.0 if self.food.x > self.head.x else 0.0
            
            return np.array([
                danger_straight, danger_right, danger_left,
                dir_up, dir_right, dir_down, dir_left,
                food_up, food_right, food_down, food_left
            ], dtype=np.float32)
            
        elif self.obs_type == "grid":
            # Float32 3D Grid representing one-hot maps of entities
            # Shape: (3, H, W)
            grid = np.zeros((3, self.grid_h, self.grid_w), dtype=np.float32)
            
            # Body (Channel 1)
            for pt in self.snake[1:]:
                gx = int(pt.x // self.block_size)
                gy = int(pt.y // self.block_size)
                if 0 <= gx < self.grid_w and 0 <= gy < self.grid_h:
                    grid[1, gy, gx] = 1.0
            # Head (Channel 0)
            hx = int(self.head.x // self.block_size)
            hy = int(self.head.y // self.block_size)
            if 0 <= hx < self.grid_w and 0 <= hy < self.grid_h:
                grid[0, hy, hx] = 1.0
            # Food (Channel 2)
            fx = int(self.food.x // self.block_size)
            fy = int(self.food.y // self.block_size)
            if 0 <= fx < self.grid_w and 0 <= fy < self.grid_h:
                # Vectorized Manhattan distance gradient map.
                # This provides a smooth slope of values that the CNN can easily follow to navigate to the food.
                max_dist = self.grid_w + self.grid_h
                y_indices, x_indices = np.indices((self.grid_h, self.grid_w))
                dists = np.abs(x_indices - fx) + np.abs(y_indices - fy)
                grid[2] = np.maximum(0.0, 1.0 - (dists / max_dist))
                
            return grid
            
        elif self.obs_type == "rgb":
            if self.display is None:
                self._init_render()
            self._render_frame()
            # Extract offscreen surface pixels
            pixels = pygame.surfarray.array3d(self.display)
            # Transpose from (W, H, C) to (H, W, C)
            return np.transpose(pixels, (1, 0, 2))

    def _get_next_point(self, dir_idx):
        x, y = self.head.x, self.head.y
        if dir_idx == 0:  # UP
            y -= self.block_size
        elif dir_idx == 1:  # RIGHT
            x += self.block_size
        elif dir_idx == 2:  # DOWN
            y += self.block_size
        elif dir_idx == 3:  # LEFT
            x -= self.block_size
        return Point(x, y)

    def _get_info(self):
        return {
            "score": self.score,
            "steps": self.total_steps,
            "steps_no_food": self.steps_no_food,
            "snake_length": len(self.snake)
        }

    def _init_render(self):
        pygame.init()
        if self.render_mode == "human":
            pygame.display.init()
            self.display = pygame.display.set_mode((self.w, self.h))
            pygame.display.set_caption("Snake RL")
            self.clock = pygame.time.Clock()
        else:
            self.display = pygame.Surface((self.w, self.h))
            
        try:
            self.font = pygame.font.Font('arial.ttf', 25)
        except FileNotFoundError:
            self.font = pygame.font.SysFont('arial', 25)

    def _render_frame(self):
        self.display.fill(self.BLACK)
        
        # Draw snake body and head
        for pt in self.snake:
            pygame.draw.rect(self.display, self.BLUE1, pygame.Rect(pt.x, pt.y, self.block_size, self.block_size))
            pygame.draw.rect(self.display, self.BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
            
        # Draw food
        pygame.draw.rect(self.display, self.RED, pygame.Rect(self.food.x, self.food.y, self.block_size, self.block_size))
        
        # Render score text
        text = self.font.render("Score: " + str(self.score), True, self.WHITE)
        self.display.blit(text, [0, 0])
        
        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def render(self):
        """
        Renders the environment according to the specified render_mode.
        """
        if self.render_mode is None:
            return None
            
        if self.display is None:
            self._init_render()
            
        self._render_frame()
        
        if self.render_mode == "rgb_array":
            pixels = pygame.surfarray.array3d(self.display)
            return np.transpose(pixels, (1, 0, 2))

    def close(self):
        """
        Closes pygame resources.
        """
        if self.display is not None:
            pygame.quit()
            self.display = None

# Automatically register environment in Gymnasium's registry
if 'Snake-v0' not in registry:
    register(
        id='Snake-v0',
        entry_point='game:SnakeEnv',
    )

if __name__ == '__main__':
    # Short test to verify logic and sanity checks
    print("Testing SnakeEnv (features observation mode)...")
    env = SnakeEnv(render_mode="rgb_array", obs_type="features")
    obs, info = env.reset()
    print(f"Features - Obs shape: {obs.shape}, sample: {obs}")
    print(f"Features - Initial info: {info}")
    
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step executed. Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    env.close()

    print("\nTesting SnakeEnv (grid observation mode)...")
    env = SnakeEnv(render_mode="rgb_array", obs_type="grid")
    obs, info = env.reset()
    print(f"Grid - Obs shape: {obs.shape}")
    print(f"Grid - Unique values in grid: {np.unique(obs)}")
    env.close()

    print("\nTesting SnakeEnv (rgb observation mode)...")
    env = SnakeEnv(render_mode="rgb_array", obs_type="rgb")
    obs, info = env.reset()
    print(f"RGB - Obs shape: {obs.shape}")
    env.close()
    
    print("\nAll sanity checks passed successfully!")
