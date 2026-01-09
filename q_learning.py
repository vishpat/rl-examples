import numpy as np
import random
from typing import Tuple, List, DefaultDict, Iterator
from collections import defaultdict
from tabulate import tabulate
import pygame
import sys

# Define the grid world
GRID_SIZE = 8 
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
        return 0 <= self.row < GRID_SIZE and 0 <= self.col < GRID_SIZE and self not in OBSTACLES

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
GOAL = State(GRID_SIZE - 2, GRID_SIZE - 1)
OBSTACLES = {State(1, 1), State(3, 1), State(5, 2), State(7, 5)}

def get_reward(state: State, next_state: State) -> int:
    if next_state == GOAL:
        return 100
    elif next_state in OBSTACLES or next_state == state:
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
    """Visualize the best action and its Q-value for each state in a grid using pygame."""
    # Initialize pygame
    pygame.init()
    
    # Constants for visualization
    CELL_SIZE = 80
    WINDOW_SIZE = GRID_SIZE * CELL_SIZE
    ARROW_SIZE = 20
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (100, 149, 237)
    GRAY = (128, 128, 128)
    LIGHT_BLUE = (173, 216, 230)
    
    # Create the window
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Q-Learning Best Actions Grid")
    
    # Font for text
    font = pygame.font.Font(None, 20)
    goal_font = pygame.font.Font(None, 24)
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Fill background
        screen.fill(WHITE)
        
        # Draw grid cells
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = j * CELL_SIZE
                y = i * CELL_SIZE
                
                # Draw cell border
                pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)
                
                state = State(i, j)
                
                if state == GOAL:
                    # Draw goal cell
                    pygame.draw.rect(screen, GREEN, (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))
                    text = goal_font.render("GOAL", True, BLACK)
                    text_rect = text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                    screen.blit(text, text_rect)
                    
                elif state in OBSTACLES:
                    # Draw obstacle cell
                    pygame.draw.rect(screen, GRAY, (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))
                    text = font.render("OBSTACLE", True, WHITE)
                    text_rect = text.get_rect(center=(x + CELL_SIZE // 2, y + CELL_SIZE // 2))
                    screen.blit(text, text_rect)
                    
                else:
                    # Draw normal cell with best action
                    best_action = max(AllowedActions, key=lambda a: q_table[state, a])
                    q_value = q_table[state, best_action]
                    
                    # Draw light background
                    pygame.draw.rect(screen, LIGHT_BLUE, (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))
                    
                    # Draw arrow for best action
                    center_x = x + CELL_SIZE // 2
                    center_y = y + CELL_SIZE // 2
                    
                    if best_action == UP:
                        # Up arrow
                        points = [
                            (center_x, center_y - ARROW_SIZE),
                            (center_x - ARROW_SIZE // 2, center_y),
                            (center_x + ARROW_SIZE // 2, center_y)
                        ]
                    elif best_action == DOWN:
                        # Down arrow
                        points = [
                            (center_x, center_y + ARROW_SIZE),
                            (center_x - ARROW_SIZE // 2, center_y),
                            (center_x + ARROW_SIZE // 2, center_y)
                        ]
                    elif best_action == LEFT:
                        # Left arrow
                        points = [
                            (center_x - ARROW_SIZE, center_y),
                            (center_x, center_y - ARROW_SIZE // 2),
                            (center_x, center_y + ARROW_SIZE // 2)
                        ]
                    elif best_action == RIGHT:
                        # Right arrow
                        points = [
                            (center_x + ARROW_SIZE, center_y),
                            (center_x, center_y - ARROW_SIZE // 2),
                            (center_x, center_y + ARROW_SIZE // 2)
                        ]
                    else:
                        points = []
                    
                    if points:
                        pygame.draw.polygon(screen, BLUE, points)
                    
                    # Draw Q-value
                    q_text = font.render(f"{q_value:.2f}", True, BLACK)
                    q_rect = q_text.get_rect(center=(center_x, y + CELL_SIZE - 15))
                    screen.blit(q_text, q_rect)
        
        # Update display
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

# Visualize the Q-table as a grid
for item in q_table:
    state, action = item
    print(f"State: {state}, Action: {action}")
# Visualize the best actions and their Q-values in a grid
visualize_best_actions_grid(q_table)