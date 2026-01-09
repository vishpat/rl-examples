import numpy as np
import random
from typing import Tuple, List, DefaultDict, Iterator
from collections import defaultdict
from tabulate import tabulate

# Define the grid world
GRID_SIZE = 3
EPSILON = 0.3
ALPHA = 0.3
GAMMA = 0.99
EPISODES = 1000 

class Action:

    def __init__(self, vertical: int, horizontal: int):
        self.vertical = vertical
        self.horizontal = horizontal

    def __eq__(self, other: 'Action') -> bool:
        return self.vertical == other.vertical and self.horizontal == other.horizontal

    def __hash__(self) -> int:
        return hash((self.vertical, self.horizontal))

    def __str__(self) -> str:
        if self.vertical == -1:
            return "^"
        elif self.vertical == 1:
            return "v"
        elif self.horizontal == 1:
            return ">"
        elif self.horizontal == -1:
            return "<"
        else:
            return "?"

    def __repr__(self) -> str:
        return self.__str__()

    def __iter__(self) -> Iterator[int]:
        return iter((self.vertical, self.horizontal))

UP = Action(-1, 0)
DOWN = Action(1, 0)
LEFT = Action(0, -1)
RIGHT = Action(0, 1)

AllowedActions = {UP, DOWN, LEFT, RIGHT}

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

    def is_valid(self) -> bool:
        return 0 <= self.row < GRID_SIZE and 0 <= self.col < GRID_SIZE and self != OBSTACLE

    def next(self, action: Action) -> 'State':
        next_state = State(self.row + action.vertical, self.col + action.horizontal)
        return next_state if next_state.is_valid() else self


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
        return tabulate(self.table.items(), headers=["State", "Action", "Q-value"], tablefmt="grid")

    def __repr__(self) -> str:
        return tabulate(self.table.items(), headers=["State", "Action", "Q-value"], tablefmt="grid")

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
    # If next_state is terminal (GOAL), there are no future rewards
    if next_state == GOAL:
        max_next_q = 0
    else:
        max_next_q = max(q_table[next_state, a] for a in AllowedActions)
    delta = ALPHA * (reward + GAMMA * max_next_q - q_table[state, action])
    q_table[state, action] += delta

START = State(0, 0)
GOAL = State(2, 2)
OBSTACLE = State(1, 1)

def get_reward(state: State, next_state: State) -> int:
    if next_state == GOAL:
        return 100
    elif next_state == OBSTACLE or next_state == state:
        return -10
    else:
        return -1

def train_agent() -> QTable:
    q_table = QTable()
    
    for _ in range(EPISODES):
        state = State(*START)
        while state != State(*GOAL):
            action = choose_action(state, q_table)
            next_state = state.next(action)
            reward = get_reward(state, next_state)
            update_q_table(q_table, state, action, reward, next_state)
            state = next_state
    
    return q_table

# Train the agent
q_table = train_agent()

def visualize_best_actions_grid(q_table: QTable) -> None:
    """Visualize the best action and its Q-value for each state in a grid."""
    print("\nBest Actions Grid:")
    header = "-" * (14 * GRID_SIZE + 1)
    print(header)

    for i in range(GRID_SIZE):
        row = "| "
        for j in range(GRID_SIZE):
            if State(i, j) == GOAL:
                cell = "   GOAL    "
            elif State(i, j) == OBSTACLE:
                cell = " OBSTACLE  "
            else:
                best_action = max(AllowedActions, key=lambda a: q_table[State(i, j), a])
                cell = f"{best_action} {q_table[State(i, j), best_action]:7.2f}  "
            row += cell + " | "
        print(row)
        print(header)

# Visualize the Q-table as a grid
for item in q_table:
    state, action = item
    print(f"State: {state}, Action: {action}")
# Visualize the best actions and their Q-values in a grid
visualize_best_actions_grid(q_table)