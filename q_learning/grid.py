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

def get_path_to_goal(start_state: State, q_table: QTable, max_steps: int = 100) -> List[State]:
    """Trace the path from start_state to GOAL using the best actions from q_table."""
    path = [start_state]
    current_state = start_state
    steps = 0
    
    while current_state != GOAL and steps < max_steps:
        # Get the best action for the current state
        best_action = max(AllowedActions, key=lambda a: q_table[current_state, a])
        # Move to the next state
        next_state = current_state.next(best_action)
        
        # Avoid infinite loops (if agent gets stuck)
        if next_state in path:
            break
            
        path.append(next_state)
        current_state = next_state
        steps += 1
    
    return path

def visualize_best_actions_grid(q_table: QTable) -> None:
    """Visualize the best action and its Q-value for each state in a grid using pygame."""
    # Initialize pygame
    pygame.init()
    
    # Constants for visualization
    CELL_SIZE = 80
    WINDOW_SIZE = GRID_SIZE * CELL_SIZE
    HEADER_HEIGHT = 60
    ARROW_SIZE = 20
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    BLUE = (100, 149, 237)
    GRAY = (128, 128, 128)
    LIGHT_BLUE = (173, 216, 230)
    YELLOW = (255, 255, 0)
    ORANGE = (255, 165, 0)
    
    # Create the window with header space
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE + HEADER_HEIGHT))
    pygame.display.set_caption("Q-Learning Best Actions Grid - Click a cell to see path to goal")
    
    # Font for text
    font = pygame.font.Font(None, 20)
    goal_font = pygame.font.Font(None, 24)
    header_font = pygame.font.Font(None, 22)
    
    # Track the selected path
    selected_path = []
    
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
                elif event.key == pygame.K_SPACE or event.key == pygame.K_c:
                    # Clear the selected path
                    selected_path = []
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Get mouse position and convert to grid coordinates
                mouse_x, mouse_y = pygame.mouse.get_pos()
                
                # Adjust for header height
                if mouse_y >= HEADER_HEIGHT:
                    grid_col = mouse_x // CELL_SIZE
                    grid_row = (mouse_y - HEADER_HEIGHT) // CELL_SIZE
                    
                    # Check if click is within grid bounds
                    if 0 <= grid_row < GRID_SIZE and 0 <= grid_col < GRID_SIZE:
                        clicked_state = State(grid_row, grid_col)
                        
                        # Only trace path if it's not an obstacle
                        if clicked_state not in OBSTACLES:
                            selected_path = get_path_to_goal(clicked_state, q_table)
        
        # Fill background
        screen.fill(WHITE)
        
        # Draw header
        pygame.draw.rect(screen, LIGHT_BLUE, (0, 0, WINDOW_SIZE, HEADER_HEIGHT))
        pygame.draw.line(screen, BLACK, (0, HEADER_HEIGHT), (WINDOW_SIZE, HEADER_HEIGHT), 2)
        
        # Draw instructions
        instruction_text = "Click on a cell to see path to goal | Press SPACE/C to clear | Press ESC to quit"
        inst_surface = header_font.render(instruction_text, True, BLACK)
        inst_rect = inst_surface.get_rect(center=(WINDOW_SIZE // 2, 20))
        screen.blit(inst_surface, inst_rect)
        
        # Draw path info if a path is selected
        if selected_path:
            path_info = f"Path length: {len(selected_path)} steps"
            info_surface = font.render(path_info, True, BLUE)
            info_rect = info_surface.get_rect(center=(WINDOW_SIZE // 2, 45))
            screen.blit(info_surface, info_rect)
        
        # Draw grid cells (offset by header height)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = j * CELL_SIZE
                y = i * CELL_SIZE + HEADER_HEIGHT
                
                # Draw cell border
                pygame.draw.rect(screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)
                
                state = State(i, j)
                
                # Check if this cell is in the selected path
                is_in_path = state in selected_path
                path_index = selected_path.index(state) if is_in_path else -1
                
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
                    
                    # Draw background - highlight if in path
                    if is_in_path:
                        if path_index == 0:
                            # Starting cell - use orange
                            bg_color = ORANGE
                        else:
                            # Path cell - use yellow
                            bg_color = YELLOW
                    else:
                        bg_color = LIGHT_BLUE
                    
                    pygame.draw.rect(screen, bg_color, (x + 2, y + 2, CELL_SIZE - 4, CELL_SIZE - 4))
                    
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
                    
                    # Draw Q-value and step number if in path
                    if is_in_path:
                        # Show step number
                        step_text = font.render(f"Step {path_index}", True, BLACK)
                        step_rect = step_text.get_rect(center=(center_x, y + 10))
                        screen.blit(step_text, step_rect)
                        
                        q_text = font.render(f"{q_value:.2f}", True, BLACK)
                        q_rect = q_text.get_rect(center=(center_x, y + CELL_SIZE - 15))
                        screen.blit(q_text, q_rect)
                    else:
                        # Just show Q-value
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