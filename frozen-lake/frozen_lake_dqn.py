
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


# -----------------------------
# Core types
# -----------------------------
class Action(IntEnum):
    LEFT = 0
    RIGHT = 1
    UP = 2
    DOWN = 3


@dataclass(frozen=True)
class EnvConfig:
    width: int = 8
    height: int = 8
    hole_prob: float = 0.18
    ensure_solvable: bool = True
    random_map_each_reset: bool = True
    max_gen_tries: int = 5000

    # "grid": (H,W) uint8 codes: 0 safe, 1 hole, 2 goal, 3 agent
    # "onehot": (3,H,W) float32 planes: [holes, goal, agent]
    obs_mode: str = "grid"

    step_reward: float = -0.01
    goal_reward: float = 1.0
    hole_reward: float = 0.0

    max_steps: Optional[int] = None
    render_mode: Optional[str] = None  # None, "ansi", "human"
    seed: Optional[int] = None


@dataclass(frozen=True)
class State:
    board: np.ndarray  # (H,W) int8: 0 safe, 1 hole, 2 goal
    start_xy: Tuple[int, int]
    goal_xy: Tuple[int, int]
    agent_xy: Tuple[int, int]
    steps: int


# -----------------------------
# Environment
# -----------------------------
class FrozenLakeRandomEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 10}

    def __init__(self, config: EnvConfig = EnvConfig()):
        super().__init__()
        self.cfg = config
        self.w = int(self.cfg.width)
        self.h = int(self.cfg.height)

        self._SAFE = 0
        self._HOLE = 1
        self._GOAL = 2
        self._AGENT = 3

        self.action_space = spaces.Discrete(4)

        if self.cfg.obs_mode == "grid":
            self.observation_space = spaces.Box(low=0, high=3, shape=(self.h, self.w), dtype=np.uint8)
        elif self.cfg.obs_mode == "onehot":
            self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, self.h, self.w), dtype=np.float32)
        else:
            raise ValueError("obs_mode must be 'grid' or 'onehot'")

        self._rng = np.random.default_rng(self.cfg.seed)
        self._state: Optional[State] = None

        # pygame state
        self._pygame_inited = False
        self._screen = None
        self._clock = None

        self._state = self._make_initial_state()

    def _neighbors(self, x: int, y: int):
        if x + 1 < self.w:
            yield x + 1, y
        if x - 1 >= 0:
            yield x - 1, y
        if y + 1 < self.h:
            yield x, y + 1
        if y - 1 >= 0:
            yield x, y - 1

    def _is_solvable(self, board: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
        from collections import deque

        sx, sy = start
        gx, gy = goal
        q = deque([(sx, sy)])
        seen = {(sx, sy)}
        while q:
            x, y = q.popleft()
            if (x, y) == (gx, gy):
                return True
            for nx, ny in self._neighbors(x, y):
                if (nx, ny) not in seen and board[ny, nx] != self._HOLE:
                    seen.add((nx, ny))
                    q.append((nx, ny))
        return False

    def _generate_map(self):
        all_cells = [(x, y) for y in range(self.h) for x in range(self.w)]
        start = all_cells[self._rng.integers(len(all_cells))]
        goal = start
        while goal == start:
            goal = all_cells[self._rng.integers(len(all_cells))]

        for _ in range(self.cfg.max_gen_tries):
            board = np.full((self.h, self.w), self._SAFE, dtype=np.int8)
            mask = self._rng.random((self.h, self.w)) < self.cfg.hole_prob
            board[mask] = self._HOLE

            sx, sy = start
            gx, gy = goal
            board[sy, sx] = self._SAFE
            board[gy, gx] = self._GOAL

            if (not self.cfg.ensure_solvable) or self._is_solvable(board, start, goal):
                return board, start, goal

        board = np.full((self.h, self.w), self._SAFE, dtype=np.int8)
        gx, gy = goal
        board[gy, gx] = self._GOAL
        return board, start, goal

    def _make_initial_state(self) -> State:
        board, start, goal = self._generate_map()
        return State(board=board, start_xy=start, goal_xy=goal, agent_xy=start, steps=0)

    def _obs_grid(self, s: State) -> np.ndarray:
        ax, ay = s.agent_xy
        obs = s.board.astype(np.uint8).copy()
        obs[ay, ax] = self._AGENT
        return obs

    def _obs_onehot(self, s: State) -> np.ndarray:
        ax, ay = s.agent_xy
        holes = (s.board == self._HOLE).astype(np.float32)
        goal = (s.board == self._GOAL).astype(np.float32)
        agent = np.zeros((self.h, self.w), dtype=np.float32)
        agent[ay, ax] = 1.0
        return np.stack([holes, goal, agent], axis=0)

    def _make_obs(self, s: State) -> np.ndarray:
        return self._obs_grid(s) if self.cfg.obs_mode == "grid" else self._obs_onehot(s)

    def _apply_action(self, agent_xy: Tuple[int, int], action: Action) -> Tuple[int, int]:
        x, y = agent_xy
        if action == Action.LEFT:
            x = max(0, x - 1)
        elif action == Action.RIGHT:
            x = min(self.w - 1, x + 1)
        elif action == Action.UP:
            y = max(0, y - 1)
        elif action == Action.DOWN:
            y = min(self.h - 1, y + 1)
        else:
            raise ValueError(f"Invalid action {action}")
        return x, y

    def _reward_done(self, board: np.ndarray, agent_xy: Tuple[int, int]) -> Tuple[float, bool]:
        x, y = agent_xy
        tile = int(board[y, x])
        if tile == self._HOLE:
            return float(self.cfg.hole_reward), True
        if tile == self._GOAL:
            return float(self.cfg.goal_reward), True
        return float(self.cfg.step_reward), False

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._state is None or self.cfg.random_map_each_reset:
            self._state = self._make_initial_state()
        else:
            s = self._state
            self._state = State(board=s.board, start_xy=s.start_xy, goal_xy=s.goal_xy, agent_xy=s.start_xy, steps=0)

        obs = self._make_obs(self._state)
        info = {"start": self._state.start_xy, "goal": self._state.goal_xy, "board": self._state.board.copy()}
        return obs, info

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        act = Action(int(action))
        s = self._state
        next_steps = s.steps + 1

        next_xy = self._apply_action(s.agent_xy, act)
        reward, terminated = self._reward_done(s.board, next_xy)

        max_steps = self.cfg.max_steps if self.cfg.max_steps is not None else self.w * self.h * 4
        truncated = (next_steps >= max_steps) and (not terminated)

        self._state = State(
            board=s.board,
            start_xy=s.start_xy,
            goal_xy=s.goal_xy,
            agent_xy=next_xy,
            steps=next_steps,
        )
        obs = self._make_obs(self._state)

        if self.cfg.render_mode is not None:
            self.render()

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self._state is None:
            return None

        if self.cfg.render_mode == "ansi":
            ax, ay = self._state.agent_xy
            lines = []
            for y in range(self.h):
                row = []
                for x in range(self.w):
                    if (x, y) == (ax, ay):
                        row.append("A")
                    else:
                        t = int(self._state.board[y, x])
                        row.append("." if t == self._SAFE else ("H" if t == self._HOLE else "G"))
                lines.append(" ".join(row))
            return "\n".join(lines)

        if self.cfg.render_mode == "human":
            import pygame

            TILE = 64
            MARGIN = 2
            W = self.w * TILE
            H = self.h * TILE

            COL_BG = (18, 18, 22)
            COL_SAFE = (180, 210, 255)
            COL_HOLE = (20, 30, 60)
            COL_GOAL = (255, 215, 90)
            COL_AGENT = (220, 60, 60)
            COL_GRID = (30, 40, 70)

            if not self._pygame_inited:
                pygame.init()
                self._screen = pygame.display.set_mode((W, H))
                pygame.display.set_caption("FrozenLakeRandomEnv (full obs)")
                self._clock = pygame.time.Clock()
                self._pygame_inited = True

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pass

            self._screen.fill(COL_BG)

            for y in range(self.h):
                for x in range(self.w):
                    t = int(self._state.board[y, x])
                    color = COL_SAFE if t == self._SAFE else (COL_HOLE if t == self._HOLE else COL_GOAL)
                    rect = pygame.Rect(
                        x * TILE + MARGIN, y * TILE + MARGIN, TILE - 2 * MARGIN, TILE - 2 * MARGIN
                    )
                    pygame.draw.rect(self._screen, color, rect, border_radius=8)
                    pygame.draw.rect(self._screen, COL_GRID, rect, width=2, border_radius=8)

            ax, ay = self._state.agent_xy
            cx = ax * TILE + TILE // 2
            cy = ay * TILE + TILE // 2
            pygame.draw.circle(self._screen, COL_AGENT, (cx, cy), TILE // 4)

            pygame.display.flip()
            self._clock.tick(self.metadata["render_fps"])

        return None

    def close(self):
        if self._pygame_inited:
            import pygame

            pygame.quit()
            self._pygame_inited = False
            self._screen = None
            self._clock = None


# -----------------------------
# SB3 wrappers + custom CNN
# -----------------------------
class GridToImageObs(gym.ObservationWrapper):
    """
    (H,W) uint8 codes 0..3 -> (H,W,1) uint8 scaled to [0,255]
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        h, w = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(h, w, 1), dtype=np.uint8)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        img = (obs.astype(np.uint16) * 85).astype(np.uint8)  # 0,85,170,255
        return img[..., None]


def make_small_grid_cnn():
    """
    Defined as a factory so torch is only imported when training/evaluating.
    """
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class SmallGridCNN(BaseFeaturesExtractor):
        """
        Works on small inputs like (C,8,8).
        Robust due to AdaptiveAvgPool.
        """

        def __init__(self, observation_space: spaces.Box, features_dim: int = 128):
            super().__init__(observation_space, features_dim)
            n_input_channels = observation_space.shape[0]  # (C,H,W)

            self.cnn = nn.Sequential(
                nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
            )

            with th.no_grad():
                sample = th.as_tensor(observation_space.sample()[None])
                if sample.dtype == th.uint8:
                    sample = sample.float() / 255.0
                else:
                    sample = sample.float()
                n_flatten = self.cnn(sample).shape[1]

            self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim),
                nn.ReLU(),
            )

        def forward(self, observations: th.Tensor) -> th.Tensor:
            if observations.dtype == th.uint8:
                observations = observations.float() / 255.0
            return self.linear(self.cnn(observations))

    return SmallGridCNN


# -----------------------------
# CLI
# -----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DQN on random FrozenLake with full-board observations (fixed small CNN).")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_env_args(sp):
        sp.add_argument("--width", type=int, default=8)
        sp.add_argument("--height", type=int, default=8)
        sp.add_argument("--hole-prob", type=float, default=0.18)
        sp.add_argument("--no-ensure-solvable", action="store_true")
        sp.add_argument("--fixed-map", action="store_true")
        sp.add_argument("--seed", type=int, default=0, help="0 => no fixed seed")

        sp.add_argument("--obs-mode", choices=["grid", "onehot"], default="grid")

        sp.add_argument("--step-reward", type=float, default=-0.01)
        sp.add_argument("--goal-reward", type=float, default=1.0)
        sp.add_argument("--hole-reward", type=float, default=0.0)
        sp.add_argument("--max-steps", type=int, default=0, help="0 => default W*H*4")

    tr = sub.add_parser("train", help="Train DQN and save model")
    add_env_args(tr)
    tr.add_argument("--timesteps", type=int, default=300_000)
    tr.add_argument("--save-path", type=str, default="dqn_frozen_lake_fullobs.zip")

    tr.add_argument("--learning-rate", type=float, default=1e-3)
    tr.add_argument("--buffer-size", type=int, default=100_000)
    tr.add_argument("--learning-starts", type=int, default=2_000)
    tr.add_argument("--batch-size", type=int, default=64)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--train-freq", type=int, default=4)
    tr.add_argument("--target-update-interval", type=int, default=1_000)
    tr.add_argument("--exploration-fraction", type=float, default=0.2)
    tr.add_argument("--exploration-final-eps", type=float, default=0.05)

    ev = sub.add_parser("eval", help="Evaluate a saved model")
    add_env_args(ev)
    ev.add_argument("--load-path", type=str, default="dqn_frozen_lake_fullobs.zip")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--render", choices=["none", "ansi", "human"], default="human")
    ev.add_argument("--deterministic", action="store_true", default=True)

    return p


def cfg_from_args(args, render_mode: Optional[str]) -> EnvConfig:
    max_steps = None if args.max_steps in (None, 0) else int(args.max_steps)
    seed = None if args.seed in (None, 0) else int(args.seed)
    random_map_each_reset = not bool(args.fixed_map)

    return EnvConfig(
        width=args.width,
        height=args.height,
        hole_prob=args.hole_prob,
        ensure_solvable=not args.no_ensure_solvable,
        random_map_each_reset=random_map_each_reset,
        obs_mode=args.obs_mode,
        step_reward=args.step_reward,
        goal_reward=args.goal_reward,
        hole_reward=args.hole_reward,
        max_steps=max_steps,
        render_mode=render_mode,
        seed=seed,
    )


def make_sb3_env(cfg: EnvConfig):
    from stable_baselines3.common.monitor import Monitor

    env = FrozenLakeRandomEnv(cfg)
    env = Monitor(env)

    if cfg.obs_mode == "grid":
        env = GridToImageObs(env)  # (H,W,1) uint8 [0,255]
    # if onehot: already (3,H,W) float32
    return env


def train_main(args):
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
    from stable_baselines3.common.callbacks import EvalCallback

    cfg = cfg_from_args(args, render_mode=None)

    # Vectorized env wrapper (DQN expects VecEnv; use 1 env)
    env = DummyVecEnv([lambda: make_sb3_env(cfg)])

    # For grid images: (H,W,1) -> transpose to (1,H,W) for PyTorch CNN
    if cfg.obs_mode == "grid":
        env = VecTransposeImage(env)

    # Eval env (same preprocessing pipeline!)
    eval_cfg = EnvConfig(**{**cfg.__dict__, "random_map_each_reset": True})
    eval_env = DummyVecEnv([lambda: make_sb3_env(eval_cfg)])
    if cfg.obs_mode == "grid":
        eval_env = VecTransposeImage(eval_env)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="best_model",
        log_path="logs",
        eval_freq=10_000,
        n_eval_episodes=10,
        deterministic=True,
    )

    SmallGridCNN = make_small_grid_cnn()
    policy_kwargs = dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        train_freq=args.train_freq,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=policy_kwargs,
    )

    model.learn(total_timesteps=args.timesteps, callback=eval_cb)
    model.save(args.save_path)

    eval_env.close()
    env.close()


def eval_main(args):
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

    render_mode = None if args.render == "none" else args.render
    cfg = cfg_from_args(args, render_mode=render_mode)

    env = DummyVecEnv([lambda: make_sb3_env(cfg)])
    if cfg.obs_mode == "grid":
        env = VecTransposeImage(env)

    model = DQN.load(args.load_path)

    for ep in range(args.episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = env.step(action)
            done = bool(dones[0])
            ep_return += float(reward[0])

            if render_mode == "ansi":
                print(env.envs[0].render())
                print("-" * 40)

        print(f"Episode {ep + 1}/{args.episodes} return: {ep_return:.3f}")

    env.close()


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "train":
        train_main(args)
    else:
        eval_main(args)
