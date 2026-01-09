import numpy as np
import random
from typing import Tuple, List, DefaultDict, Iterator
from collections import defaultdict

# Define the grid world
GRID_SIZE = 3
START = (0, 0)
GOAL = (2, 2)
OBSTACLE = (1, 1)

EPSILON = 0.3
ALPHA = 0.3
GAMMA = 0.99
EPISODES = 10000

class Action:

    def __init__(self, vertical: int, horizontal: int):
        self.vertical = vertical
        self.horizontal = horizontal

    def __eq__(self, other: 'Action') -> bool:
        return self.vertical == other.vertical and self.horizontal == other.horizontal

    def __hash__(self) -> int:
        return hash((self.vertical, self.horizontal))

    def __str__(self) -> str:
        return f"({self.vertical}, {self.horizontal})"

    def __repr__(self) -> str:
        return f"Action(vertical={self.vertical}, horizontal={self.horizontal})"

    def __iter__(self) -> Iterator[int]:
        return iter((self.vertical, self.horizontal))

AllowedActions = {
    Action(0, 1),
    Action(0, -1),
    Action(1, 0),
    Action(-1, 0),
}
class State:
    def __init__(self, row: int, col: int):
        self.row = row
        self.col = col

    def __eq__(self, other: 'State') -> bool:
        return self.row == other.row and self.col == other.col

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __str__(self) -> str:
        return f"({self.row}, {self.col})"
    
    def __repr__(self) -> str:
        return f"State(row={self.row}, col={self.col})"
    
    def __iter__(self) -> Iterator[int]:
        return iter((self.row, self.col))

    def next(self, action: Action) -> 'State':
        return State(self.row + action.horizontal, self.col + action.vertical)



class QTable:
    def __init__(self):
        self.table = defaultdict(lambda: defaultdict(float))

    def __getitem__(self, key: Tuple[State, Action]) -> float:
        state, action = key
        return self.table[state][action]

    def __setitem__(self, key: Tuple[State, Action], value: float) -> None:
        state, action = key
        self.table[state][action] = value

    def __contains__(self, key: Tuple[State, Action]) -> bool:
        state, action = key
        return state in self.table and action in self.table[state]

    def __iter__(self) -> Iterator[Tuple[State, Action, float]]:
        return iter(self.table.items())

    def __len__(self) -> int:
        return len(self.table)

    def __str__(self) -> str:
        return str(self.table)

    def __repr__(self) -> str:
        return f"QTable(table={self.table})"

    def __hash__(self) -> int:
        return hash(tuple(self.table.items()))

    def __eq__(self, other: 'QTable') -> bool:
        return self.table == other.table

    def __ne__(self, other: 'QTable') -> bool:
        return self.table != other.table

    def __copy__(self) -> 'QTable':
        return QTable(self.table.copy())

    def __deepcopy__(self, memo: dict) -> 'QTable':
        return QTable(copy.deepcopy(self.table, memo))

def choose_action(state: State, q_table: QTable) -> Action:
    if random.uniform(0, 1) < EPSILON:
        return random.choice(list(AllowedActions))
    else:
        return max(list(AllowedActions), key=lambda action: q_table[state, action])

def update_q_table(q_table: QTable, state: State, action: Action, 
                   reward: int, next_state: State) -> None:
    max_next_q = max(q_table[next_state, a] for a in AllowedActions)
    q_table[state, action] += ALPHA * (reward + GAMMA * max_next_q - q_table[state, action])

def train_agent() -> QTable:
    q_table = QTable()
    
    for _ in range(EPISODES):
        state = State(*START)
        while state != State(*GOAL):
            action = choose_action(state, q_table)
            next_state = state.next(action)
            print(f"State: {state}, Action: {action}, Next State: {next_state}")
            reward = 1 if next_state == State(*GOAL) else -1
            update_q_table(q_table, state, action, reward, next_state)
            state = next_state
    
    return q_table

# Train the agent
q_table = train_agent()

def visualize_q_table_as_grid(q_table: QTable) -> None:
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
                    q_value = q_table[State(i, j)][action_idx]
                    cell = f" {q_value:9.2f} "
                row += cell + "|"
        print(row)
        print("-" * len(header))

def visualize_best_actions_grid(q_table: QTable) -> None:
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
                best_action_idx = max(q_table[State(i, j)].items(), key=lambda x: x[1])[0]
                best_q_value = q_table[State(i, j)][best_action_idx]
                cell = f"{action_symbols[best_action_idx]}:{best_q_value:7.2f}  "
            row += cell + " | "
        print(row)
        print(header)

# Visualize the Q-table as a grid
visualize_q_table_as_grid(q_table)

# Visualize the best actions and their Q-values in a grid
visualize_best_actions_grid(q_table)