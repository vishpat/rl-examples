# frozen_lake_random_env_cli.py
from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import gymnasium as gym
from gymnasium import spaces


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

    obs_mode: str = "state"  # "state" or "xy"

    step_reward: float = -0.01
    goal_reward: float = 1.0
    hole_reward: float = 0.0

    max_steps: Optional[int] = None
    render_mode: Optional[str] = None  # None, "ansi", "human"
    seed: Optional[int] = None


@dataclass(frozen=True)
class State:
    board: np.ndarray
    start_xy: Tuple[int, int]
    goal_xy: Tuple[int, int]
    agent_xy: Tuple[int, int]
    steps: int


@dataclass(frozen=True)
class Observation:
    value: Union[int, np.ndarray]


@dataclass(frozen=True)
class Reward:
    value: float


@dataclass(frozen=True)
class Transition:
    obs: Observation
    reward: Reward
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


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

        self.action_space = spaces.Discrete(4)
        if self.cfg.obs_mode == "xy":
            self.observation_space = spaces.MultiDiscrete([self.w, self.h])
        else:
            self.observation_space = spaces.Discrete(self.w * self.h)

        self._rng = np.random.default_rng(self.cfg.seed)
        self._state: Optional[State] = None

        # pygame state (optional)
        self._pygame_inited = False
        self._screen = None
        self._clock = None

        self._state = self._make_initial_state()

    # -------- helpers --------
    def _xy_to_state_id(self, x: int, y: int) -> int:
        return y * self.w + x

    def _make_observation(self, agent_xy: Tuple[int, int]) -> Observation:
        x, y = agent_xy
        if self.cfg.obs_mode == "xy":
            return Observation(np.array([x, y], dtype=np.int64))
        return Observation(int(self._xy_to_state_id(x, y)))

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

    def _generate_map(self) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
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

        # fallback: no holes
        board = np.full((self.h, self.w), self._SAFE, dtype=np.int8)
        gx, gy = goal
        board[gy, gx] = self._GOAL
        return board, start, goal

    def _make_initial_state(self) -> State:
        board, start, goal = self._generate_map()
        return State(board=board, start_xy=start, goal_xy=goal, agent_xy=start, steps=0)

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
            raise ValueError(f"Invalid action: {action}")
        return x, y

    def _compute_reward_and_done(self, board: np.ndarray, agent_xy: Tuple[int, int]) -> Tuple[Reward, bool]:
        x, y = agent_xy
        tile = int(board[y, x])
        if tile == self._HOLE:
            return Reward(self.cfg.hole_reward), True
        if tile == self._GOAL:
            return Reward(self.cfg.goal_reward), True
        return Reward(self.cfg.step_reward), False

    # -------- gym API --------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if self._state is None or self.cfg.random_map_each_reset:
            self._state = self._make_initial_state()
        else:
            s = self._state
            self._state = State(
                board=s.board,
                start_xy=s.start_xy,
                goal_xy=s.goal_xy,
                agent_xy=s.start_xy,
                steps=0,
            )

        obs = self._make_observation(self._state.agent_xy)
        info = {"start": self._state.start_xy, "goal": self._state.goal_xy, "board": self._state.board.copy()}
        return obs.value, info

    def step(self, action: int):
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        act = Action(int(action))
        s = self._state
        next_steps = s.steps + 1

        next_xy = self._apply_action(s.agent_xy, act)
        reward, terminated = self._compute_reward_and_done(s.board, next_xy)

        max_steps = self.cfg.max_steps if self.cfg.max_steps is not None else self.w * self.h * 4
        truncated = (next_steps >= max_steps) and (not terminated)

        self._state = State(
            board=s.board,
            start_xy=s.start_xy,
            goal_xy=s.goal_xy,
            agent_xy=next_xy,
            steps=next_steps,
        )

        obs = self._make_observation(self._state.agent_xy)
        transition = Transition(obs=obs, reward=reward, terminated=terminated, truncated=truncated, info={})

        if self.cfg.render_mode is not None:
            self.render()

        return (
            transition.obs.value,
            transition.reward.value,
            transition.terminated,
            transition.truncated,
            transition.info,
        )

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
                pygame.display.set_caption("FrozenLakeRandomEnv")
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
                        x * TILE + MARGIN,
                        y * TILE + MARGIN,
                        TILE - 2 * MARGIN,
                        TILE - 2 * MARGIN,
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


# ------------------------------
# CLI: train / eval
# ------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="FrozenLakeRandomEnv: train/eval with Stable-Baselines3")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Shared env args
    def add_env_args(sp):
        sp.add_argument("--width", type=int, default=8)
        sp.add_argument("--height", type=int, default=8)
        sp.add_argument("--hole-prob", type=float, default=0.18)
        sp.add_argument("--no-ensure-solvable", action="store_true")
        sp.add_argument("--random-map-each-reset", action="store_true", default=True)
        sp.add_argument("--fixed-map", action="store_true", help="Do not regenerate map on reset (overrides --random-map-each-reset)")
        sp.add_argument("--obs-mode", choices=["state", "xy"], default="state")
        sp.add_argument("--step-reward", type=float, default=-0.01)
        sp.add_argument("--goal-reward", type=float, default=1.0)
        sp.add_argument("--hole-reward", type=float, default=0.0)
        sp.add_argument("--max-steps", type=int, default=0, help="0 means default (W*H*4)")
        sp.add_argument("--seed", type=int, default=0, help="0 means no fixed seed")

    train = sub.add_parser("train", help="Train a policy and save it")
    add_env_args(train)
    train.add_argument("--algo", choices=["ppo"], default="ppo")
    train.add_argument("--timesteps", type=int, default=200_000)
    train.add_argument("--n-envs", type=int, default=8)
    train.add_argument("--save-path", type=str, default="ppo_frozen_lake_random.zip")
    train.add_argument("--policy", type=str, default="MlpPolicy")
    train.add_argument("--n-steps", type=int, default=256)
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--gamma", type=float, default=0.99)

    ev = sub.add_parser("eval", help="Load a policy and run episodes")
    add_env_args(ev)
    ev.add_argument("--load-path", type=str, default="ppo_frozen_lake_random.zip")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--render", choices=["none", "ansi", "human"], default="human")
    ev.add_argument("--deterministic", action="store_true", default=True)

    return p


def cfg_from_args(args, render_mode: Optional[str] = None) -> EnvConfig:
    max_steps = None if getattr(args, "max_steps", 0) in (None, 0) else int(args.max_steps)
    seed = None if getattr(args, "seed", 0) in (None, 0) else int(args.seed)

    random_map_each_reset = True
    if getattr(args, "fixed_map", False):
        random_map_each_reset = False
    else:
        random_map_each_reset = bool(getattr(args, "random_map_each_reset", True))

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


def train_main(args):
    from stable_baselines3 import DQN
    from stable_baselines3.common.env_util import make_vec_env

    cfg = cfg_from_args(args, render_mode=None)

    def make_env():
        return FrozenLakeRandomEnv(cfg)

    env = make_vec_env(make_env, n_envs=args.n_envs)

    model = DQN(
        args.policy,
        env,
        verbose=1,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(args.save_path)
    env.close()


def eval_main(args):
    from stable_baselines3 import DQN

    render_mode = None if args.render == "none" else args.render
    cfg = cfg_from_args(args, render_mode=render_mode)
    env = FrozenLakeRandomEnv(cfg)

    model = DQN.load(args.load_path)

    for ep in range(args.episodes):
        obs, info = env.reset()
        terminated = truncated = False
        ep_return = 0.0
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            ep_return += float(reward)
            if render_mode == "ansi":
                print(env.render())
                print("-" * 40)
        print(f"Episode {ep + 1}/{args.episodes} return: {ep_return:.3f}")

    env.close()


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "train":
        train_main(args)
    elif args.cmd == "eval":
        eval_main(args)
    else:
        raise RuntimeError("Unknown command")
