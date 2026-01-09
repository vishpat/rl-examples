import numpy as np
import random
from typing import Tuple, List

# Define the grid world
GRID_SIZE = 3
START = (0, 0)
GOAL = (2, 2)
OBSTACLE = (1, 1)

EPSILON = 0.3
ALPHA = 0.3
GAMMA = 0.99
EPISODES = 10000

# Define actions
ACTIONS = [
    (-1, 0),  # up
    (0, 1),   # right
    (1, 0),   # down
    (0, -1)   # left
]

def is_valid_state(state: Tuple[int, int]) -> bool:
    return (0 <= state[0] < GRID_SIZE and 
            0 <= state[1] < GRID_SIZE and 
            state != OBSTACLE)


def get_next_state(state: Tuple[int, int], action: Tuple[int, int]) -> Tuple[int, int]:
    next_state = (state[0] + action[0], state[1] + action[1])
    return next_state if is_valid_state(next_state) else state

def get_reward(state: Tuple[int, int], next_state: Tuple[int, int]) -> int:
    if next_state == GOAL:
        return 100
    elif next_state == OBSTACLE or next_state == state:  # Penalize hitting walls or obstacle
        return -10
    else:
        return -1  # Small penalty for each step to encourage shortest path

def choose_action(state: Tuple[int, int], q_table: np.ndarray) -> Tuple[int, int]:
    if random.uniform(0, 1) < EPSILON:
        return random.choice(ACTIONS)
    else:
        return ACTIONS[np.argmax(q_table[state])]

def update_q_table(q_table: np.ndarray, state: Tuple[int, int], action: Tuple[int, int], 
                   reward: int, next_state: Tuple[int, int]) -> None:
    action_idx = ACTIONS.index(action)
    q_table[state][action_idx] += ALPHA * (reward + GAMMA * np.max(q_table[next_state]) - q_table[state][action_idx])

def train_agent() -> np.ndarray:
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
    
    for _ in range(EPISODES):
        state = START
        while state != GOAL:
            action = choose_action(state, q_table)
            next_state = get_next_state(state, action)
            reward = get_reward(state, next_state)
            update_q_table(q_table, state, action, reward, next_state)
            state = next_state
    
    return q_table

# Train the agent
q_table = train_agent()

def visualize_q_table_as_grid(q_table: np.ndarray) -> None:
    """Visualize the Q-table as a grid with all action values for each state."""
    action_symbols = ['^', '>', 'v', '<']
    
    print("\nDetailed Q-table Grid:")
    
    # Header
    header = "   |" + "|".join(f"   ({i},{j})   " for i in range(GRID_SIZE) for j in range(GRID_SIZE)) + "|"
    print(header)
    print("-" * len(header))

    for action_idx, action_symbol in enumerate(action_symbols):
        row = f" {action_symbol} |"
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if (i, j) == GOAL:
                    cell = "   GOAL    "
                elif (i, j) == OBSTACLE:
                    cell = " OBSTACLE  "
                else:
                    q_value = q_table[i, j, action_idx]
                    cell = f" {q_value:9.2f} "
                row += cell + "|"
        print(row)
        print("-" * len(header))

def visualize_best_actions_grid(q_table: np.ndarray) -> None:
    """Visualize the best action and its Q-value for each state in a grid."""
    action_symbols = ['^', '>', 'v', '<']
    
    print("\nBest Actions Grid:")
    header = "-" * (14 * GRID_SIZE + 1)
    print(header)

    for i in range(GRID_SIZE):
        row = "| "
        for j in range(GRID_SIZE):
            if (i, j) == GOAL:
                cell = "   GOAL    "
            elif (i, j) == OBSTACLE:
                cell = " OBSTACLE  "
            else:
                best_action_idx = np.argmax(q_table[i, j])
                best_q_value = q_table[i, j, best_action_idx]
                cell = f"{action_symbols[best_action_idx]}:{best_q_value:7.2f}  "
            row += cell + " | "
        print(row)
        print(header)

# Visualize the Q-table as a grid
visualize_q_table_as_grid(q_table)

# Visualize the best actions and their Q-values in a grid
visualize_best_actions_grid(q_table)