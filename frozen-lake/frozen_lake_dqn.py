
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from enum import IntEnum
from pathlib import Path
from typing import Dict, Optional, Tuple

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

    # "onehot": (4,H,W) planes: [safe, holes, goal, agent]
    # "grid": (1,H,W) uint8 codes: 0 safe, 1 hole, 2 goal, 3 agent
    obs_mode: str = "onehot"

    step_reward: float = -0.01
    goal_reward: float = 1.0
    hole_reward: float = -1.0
    revisit_reward: float = -0.05
    shaping_reward_scale: float = 0.02

    max_steps: Optional[int] = 20 
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
            self.observation_space = spaces.Box(
                low=0,
                high=3,
                shape=(1, self.h, self.w),
                dtype=np.uint8,
            )
        elif self.cfg.obs_mode == "onehot":
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4, self.h, self.w),
                dtype=np.float32,
            )
        else:
            raise ValueError("obs_mode must be 'grid' or 'onehot'")

        self._rng = np.random.default_rng(self.cfg.seed)
        self._state: Optional[State] = None
        self._visited: set[Tuple[int, int]] = set()

        # pygame state
        self._pygame_inited = False
        self._screen = None
        self._clock = None

        self._state = self._make_initial_state()
        self._init_visited()

    def _init_visited(self) -> None:
        if self._state is None:
            self._visited = set()
        else:
            self._visited = {self._state.agent_xy}

    def set_curriculum(self, *, hole_prob: float, random_map_each_reset: bool) -> None:
        self.cfg = replace(
            self.cfg,
            hole_prob=float(hole_prob),
            random_map_each_reset=bool(random_map_each_reset),
        )

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
        return self._shortest_path_distance(board, start, goal) is not None

    def _shortest_path_distance(
        self,
        board: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
    ) -> Optional[int]:
        from collections import deque

        sx, sy = start
        gx, gy = goal
        if board[sy, sx] == self._HOLE:
            return None

        q = deque([(sx, sy, 0)])
        seen = {(sx, sy)}
        while q:
            x, y, dist = q.popleft()
            if (x, y) == (gx, gy):
                return dist
            for nx, ny in self._neighbors(x, y):
                if (nx, ny) not in seen and board[ny, nx] != self._HOLE:
                    seen.add((nx, ny))
                    q.append((nx, ny, dist + 1))
        return None

    def _generate_map(self):
        start = (0, 0)
        goal = (self.w - 1, self.h - 1)
        sx, sy = start
        gx, gy = goal

        for _ in range(self.cfg.max_gen_tries):
            board = np.full((self.h, self.w), self._SAFE, dtype=np.int8)
            mask = self._rng.random((self.h, self.w)) < self.cfg.hole_prob
            mask[sy, sx] = False
            mask[gy, gx] = False
            board[mask] = self._HOLE
            board[gy, gx] = self._GOAL

            if (not self.cfg.ensure_solvable) or self._is_solvable(board, start, goal):
                return board, start, goal

        board = np.full((self.h, self.w), self._SAFE, dtype=np.int8)
        board[gy, gx] = self._GOAL
        return board, start, goal

    def _make_initial_state(self) -> State:
        board, start, goal = self._generate_map()
        return State(board=board, start_xy=start, goal_xy=goal, agent_xy=start, steps=0)

    def _obs_grid_frame(self, s: State) -> np.ndarray:
        ax, ay = s.agent_xy
        obs = s.board.astype(np.uint8).copy()
        obs[ay, ax] = self._AGENT
        return obs

    def _obs_onehot_frame(self, grid_frame: np.ndarray) -> np.ndarray:
        safe = (grid_frame == self._SAFE).astype(np.float32)
        holes = (grid_frame == self._HOLE).astype(np.float32)
        goal = (grid_frame == self._GOAL).astype(np.float32)
        agent = (grid_frame == self._AGENT).astype(np.float32)
        return np.stack([safe, holes, goal, agent], axis=0)

    def _obs_grid(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("No state available for observation.")
        return self._obs_grid_frame(self._state)[None, ...]

    def _obs_onehot(self) -> np.ndarray:
        if self._state is None:
            raise RuntimeError("No state available for observation.")
        return self._obs_onehot_frame(self._obs_grid_frame(self._state))

    def _make_obs(self) -> np.ndarray:
        return self._obs_grid() if self.cfg.obs_mode == "grid" else self._obs_onehot()

    def _terminal_info(
        self,
        board: np.ndarray,
        agent_xy: Tuple[int, int],
        terminated: bool,
        truncated: bool,
        *,
        revisited: bool = False,
    ) -> Dict[str, bool]:
        ax, ay = agent_xy
        tile = int(board[ay, ax])
        return {
            "is_success": bool(terminated and tile == self._GOAL),
            "fell_in_hole": bool(terminated and tile == self._HOLE),
            "timed_out": bool(truncated and not revisited),
            "revisited": bool(revisited),
        }

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

    def _shape_reward(self, board: np.ndarray, old_xy: Tuple[int, int], new_xy: Tuple[int, int]) -> float:
        if self.cfg.shaping_reward_scale <= 0:
            return 0.0

        old_dist = self._shortest_path_distance(board, old_xy, self._state.goal_xy if self._state else new_xy)
        new_dist = self._shortest_path_distance(board, new_xy, self._state.goal_xy if self._state else new_xy)
        if old_dist is None or new_dist is None:
            return 0.0
        return float(self.cfg.shaping_reward_scale) * float(old_dist - new_dist)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._state is None or self.cfg.random_map_each_reset:
            self._state = self._make_initial_state()
        else:
            s = self._state
            self._state = State(board=s.board, start_xy=s.start_xy, goal_xy=s.goal_xy, agent_xy=s.start_xy, steps=0)

        self._init_visited()
        obs = self._make_obs()
        info = {"start": self._state.start_xy, "goal": self._state.goal_xy, "board": self._state.board.copy()}
        return obs, info

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        act = Action(int(action))
        s = self._state
        next_steps = s.steps + 1

        next_xy = self._apply_action(s.agent_xy, act)
        revisited = next_xy in self._visited
        reward, terminated = self._reward_done(s.board, next_xy)
        if not terminated:
            reward += self._shape_reward(s.board, s.agent_xy, next_xy)
        if revisited:
            reward += float(self.cfg.revisit_reward)
        self._visited.add(next_xy)

        max_steps = self.cfg.max_steps if self.cfg.max_steps is not None else self.w * self.h * 4
        truncated = revisited or ((next_steps >= max_steps) and (not terminated))

        self._state = State(
            board=s.board,
            start_xy=s.start_xy,
            goal_xy=s.goal_xy,
            agent_xy=next_xy,
            steps=next_steps,
        )
        obs = self._make_obs()
        info = self._terminal_info(s.board, next_xy, terminated, truncated, revisited=revisited)

        if self.cfg.render_mode is not None:
            self.render()

        return obs, reward, terminated, truncated, info

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
# SB3 helpers + custom CNN
# -----------------------------
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

        sp.add_argument("--obs-mode", choices=["grid", "onehot"], default="onehot")

        sp.add_argument("--step-reward", type=float, default=-0.01)
        sp.add_argument("--goal-reward", type=float, default=1.0)
        sp.add_argument("--hole-reward", type=float, default=-1.0)
        sp.add_argument("--revisit-reward", type=float, default=-0.2)
        sp.add_argument("--shaping-reward-scale", type=float, default=0.02)
        sp.add_argument("--max-steps", type=int, default=0, help="0 => default W*H*4")

    tr = sub.add_parser("train", help="Train DQN and save model")
    add_env_args(tr)
    tr.add_argument("--timesteps", type=int, default=100_000)
    tr.add_argument("--save-path", type=str, default="dqn_frozen_lake_fullobs.zip")

    tr.add_argument("--learning-rate", type=float, default=1e-3)
    tr.add_argument("--buffer-size", type=int, default=200_000)
    tr.add_argument("--learning-starts", type=int, default=5_000)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--gamma", type=float, default=0.99)
    tr.add_argument("--train-freq", type=int, default=1)
    tr.add_argument("--gradient-steps", type=int, default=1)
    tr.add_argument("--target-update-interval", type=int, default=500)
    tr.add_argument("--exploration-fraction", type=float, default=0.4)
    tr.add_argument("--exploration-initial-eps", type=float, default=1.0)
    tr.add_argument("--exploration-final-eps", type=float, default=0.05)
    tr.add_argument("--eval-freq", type=int, default=10_000)
    tr.add_argument("--eval-episodes", type=int, default=20)
    tr.add_argument("--eval-seed", type=int, default=10_000)
    tr.add_argument("--best-model-dir", type=str, default="best_model")
    tr.add_argument("--no-curriculum", action="store_true")

    ev = sub.add_parser("eval", help="Evaluate a saved model")
    add_env_args(ev)
    ev.add_argument("--load-path", type=str, default="dqn_frozen_lake_fullobs.zip")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--eval-seed", type=int, default=10_000)
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
        revisit_reward=args.revisit_reward,
        shaping_reward_scale=args.shaping_reward_scale,
        max_steps=max_steps,
        render_mode=render_mode,
        seed=seed,
    )


def make_sb3_env(cfg: EnvConfig):
    from stable_baselines3.common.monitor import Monitor

    env = FrozenLakeRandomEnv(cfg)
    env = Monitor(env)
    return env


def evaluate_dqn_model(model, cfg: EnvConfig, *, episodes: int, seed_start: int, deterministic: bool) -> Dict[str, float]:
    env = make_sb3_env(replace(cfg, render_mode=None, random_map_each_reset=True))
    returns = []
    lengths = []
    successes = 0
    holes = 0
    timeouts = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed_start + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        last_info: Dict[str, bool] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_len += 1
            last_info = info

        returns.append(ep_return)
        lengths.append(ep_len)
        successes += int(last_info.get("is_success", False))
        holes += int(last_info.get("fell_in_hole", False))
        timeouts += int(last_info.get("timed_out", False))

    env.close()
    return {
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_length": float(np.mean(lengths)) if lengths else 0.0,
        "success_rate": successes / max(episodes, 1),
        "hole_rate": holes / max(episodes, 1),
        "timeout_rate": timeouts / max(episodes, 1),
    }


def make_training_callback(args, cfg: EnvConfig):
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainingDiagnosticsCallback(BaseCallback):
        def __init__(self):
            super().__init__(verbose=0)
            self.best_success_rate = -1.0
            self.best_mean_return = -float("inf")
            self.stage_index = -1
            target_hole_prob = float(cfg.hole_prob)
            self.curriculum = [
                (0.00, 0.00, False),
                (0.25, target_hole_prob * 0.33, True),
                (0.50, target_hole_prob * 0.66, True),
                (0.75, target_hole_prob, True),
            ]

        def _on_step(self) -> bool:
            if not args.no_curriculum:
                self._maybe_update_curriculum()

            if self.n_calls == 1 or self.n_calls % int(args.eval_freq) == 0:
                metrics = evaluate_dqn_model(
                    self.model,
                    cfg,
                    episodes=int(args.eval_episodes),
                    seed_start=int(args.eval_seed),
                    deterministic=True,
                )
                for key, value in metrics.items():
                    self.logger.record(f"fixed_eval/{key}", value)
                print(
                    "fixed_eval "
                    f"steps={self.num_timesteps} "
                    f"success={metrics['success_rate']:.3f} "
                    f"return={metrics['mean_return']:.3f} "
                    f"len={metrics['mean_length']:.1f} "
                    f"holes={metrics['hole_rate']:.3f} "
                    f"timeouts={metrics['timeout_rate']:.3f}"
                )

                is_better = (
                    metrics["success_rate"] > self.best_success_rate
                    or (
                        metrics["success_rate"] == self.best_success_rate
                        and metrics["mean_return"] > self.best_mean_return
                    )
                )
                if is_better:
                    self.best_success_rate = metrics["success_rate"]
                    self.best_mean_return = metrics["mean_return"]
                    best_dir = Path(args.best_model_dir)
                    best_dir.mkdir(parents=True, exist_ok=True)
                    self.model.save(best_dir / "best_model")
            return True

        def _maybe_update_curriculum(self) -> None:
            progress = self.num_timesteps / max(int(args.timesteps), 1)
            next_stage = max(i for i, (threshold, _, _) in enumerate(self.curriculum) if progress >= threshold)
            if next_stage == self.stage_index:
                return

            self.stage_index = next_stage
            _, hole_prob, random_maps = self.curriculum[next_stage]
            self.training_env.env_method(
                "set_curriculum",
                hole_prob=hole_prob,
                random_map_each_reset=random_maps,
            )
            print(
                "curriculum "
                f"stage={next_stage + 1}/{len(self.curriculum)} "
                f"hole_prob={hole_prob:.3f} "
                f"random_maps={random_maps}"
            )

    return TrainingDiagnosticsCallback()


def train_main(args):
    from stable_baselines3 import DQN
    from stable_baselines3.common.vec_env import DummyVecEnv

    cfg = cfg_from_args(args, render_mode=None)
    train_cfg = cfg
    if not args.no_curriculum:
        train_cfg = replace(cfg, hole_prob=0.0, random_map_each_reset=False)

    # Vectorized env wrapper (DQN expects VecEnv; use 1 env)
    env = DummyVecEnv([lambda: make_sb3_env(train_cfg)])

    SmallGridCNN = make_small_grid_cnn()
    policy_kwargs = dict(
        features_extractor_class=SmallGridCNN,
        features_extractor_kwargs=dict(features_dim=128),
        normalize_images=False,
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
        gradient_steps=args.gradient_steps,
        target_update_interval=args.target_update_interval,
        exploration_fraction=args.exploration_fraction,
        exploration_initial_eps=args.exploration_initial_eps,
        exploration_final_eps=args.exploration_final_eps,
        policy_kwargs=policy_kwargs,
    )

    callback = make_training_callback(args, cfg)
    model.learn(total_timesteps=args.timesteps, callback=callback)
    model.save(args.save_path)

    env.close()


def eval_main(args):
    from stable_baselines3 import DQN

    render_mode = None if args.render == "none" else args.render
    cfg = cfg_from_args(args, render_mode=render_mode)

    env = make_sb3_env(cfg)
    model = DQN.load(args.load_path)
    successes = 0
    holes = 0
    timeouts = 0
    returns = []
    lengths = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.eval_seed + ep)
        done = False
        ep_return = 0.0
        ep_len = 0
        info: Dict[str, bool] = {}

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = bool(terminated or truncated)
            ep_return += float(reward)
            ep_len += 1

            if render_mode == "ansi":
                print(env.render())
                print("-" * 40)

        successes += int(info.get("is_success", False))
        holes += int(info.get("fell_in_hole", False))
        timeouts += int(info.get("timed_out", False))
        returns.append(ep_return)
        lengths.append(ep_len)
        print(
            f"Episode {ep + 1}/{args.episodes} "
            f"return={ep_return:.3f} "
            f"len={ep_len} "
            f"success={info.get('is_success', False)} "
            f"hole={info.get('fell_in_hole', False)} "
            f"timeout={info.get('timed_out', False)}"
        )

    print(
        "Summary "
        f"mean_return={np.mean(returns):.3f} "
        f"mean_len={np.mean(lengths):.1f} "
        f"success_rate={successes / max(args.episodes, 1):.3f} "
        f"hole_rate={holes / max(args.episodes, 1):.3f} "
        f"timeout_rate={timeouts / max(args.episodes, 1):.3f}"
    )

    env.close()


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.cmd == "train":
        train_main(args)
    else:
        eval_main(args)

