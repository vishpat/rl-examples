import pygame
import random
import sys

# ----------------------------
# Frozen Lake (random layout)
# ----------------------------

# Grid settings
GRID_W, GRID_H = 8, 8          # size of the lake
TILE = 64                      # tile size in pixels
MARGIN = 2

# Random generation settings
HOLE_PROB = 0.18               # chance a tile is a hole (tune this)
MAX_TRIES = 5000               # attempts to create a solvable board

# Colors
COL_BG = (18, 18, 22)
COL_SAFE = (180, 210, 255)     # ice
COL_HOLE = (20, 30, 60)        # hole
COL_START = (90, 220, 110)     # start
COL_GOAL = (255, 215, 90)      # goal
COL_PLAYER = (220, 60, 60)     # player
COL_GRID = (30, 40, 70)
COL_TEXT = (240, 240, 245)

# Tile types
SAFE = "."
HOLE = "H"
START = "S"
GOAL = "G"


def neighbors(x, y):
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_W and 0 <= ny < GRID_H:
            yield nx, ny


def is_solvable(board, start, goal):
    # BFS from start to goal through non-hole tiles
    from collections import deque
    sx, sy = start
    gx, gy = goal
    q = deque([(sx, sy)])
    seen = set([(sx, sy)])
    while q:
        x, y = q.popleft()
        if (x, y) == (gx, gy):
            return True
        for nx, ny in neighbors(x, y):
            if (nx, ny) not in seen and board[ny][nx] != HOLE:
                seen.add((nx, ny))
                q.append((nx, ny))
    return False


def generate_board():
    # Choose random start and goal positions (distinct)
    cells = [(x, y) for y in range(GRID_H) for x in range(GRID_W)]
    start = random.choice(cells)
    goal = random.choice([c for c in cells if c != start])

    # Try to generate a solvable random hole layout
    for _ in range(MAX_TRIES):
        board = [[SAFE for _ in range(GRID_W)] for _ in range(GRID_H)]
        sx, sy = start
        gx, gy = goal
        board[sy][sx] = START
        board[gy][gx] = GOAL

        for y in range(GRID_H):
            for x in range(GRID_W):
                if (x, y) in (start, goal):
                    continue
                if random.random() < HOLE_PROB:
                    board[y][x] = HOLE

        # Ensure solvable
        if is_solvable(board, start, goal):
            return board, start, goal

    # If fails, fall back to a board with no holes
    board = [[SAFE for _ in range(GRID_W)] for _ in range(GRID_H)]
    sx, sy = start
    gx, gy = goal
    board[sy][sx] = START
    board[gy][gx] = GOAL
    return board, start, goal


def draw_board(screen, board, player, font, msg=None):
    screen.fill(COL_BG)

    # draw tiles
    for y in range(GRID_H):
        for x in range(GRID_W):
            t = board[y][x]
            if t == SAFE:
                color = COL_SAFE
            elif t == HOLE:
                color = COL_HOLE
            elif t == START:
                color = COL_START
            elif t == GOAL:
                color = COL_GOAL
            else:
                color = COL_SAFE

            px = x * TILE
            py = y * TILE
            rect = pygame.Rect(px + MARGIN, py + MARGIN, TILE - 2 * MARGIN, TILE - 2 * MARGIN)
            pygame.draw.rect(screen, color, rect, border_radius=8)
            pygame.draw.rect(screen, COL_GRID, rect, width=2, border_radius=8)

    # draw player
    px, py = player
    cx = px * TILE + TILE // 2
    cy = py * TILE + TILE // 2
    pygame.draw.circle(screen, COL_PLAYER, (cx, cy), TILE // 4)

    # draw text
    if msg:
        surf = font.render(msg, True, COL_TEXT)
        screen.blit(surf, (8, GRID_H * TILE + 8))

    pygame.display.flip()


def main():
    pygame.init()
    pygame.display.set_caption("Frozen Lake - Random each start (R to regenerate)")

    W = GRID_W * TILE
    H = GRID_H * TILE + 40
    screen = pygame.display.set_mode((W, H))
    font = pygame.font.SysFont(None, 24)

    board, start, goal = generate_board()
    player = list(start)
    done = False
    status = "Arrows move. Reach goal (gold). Avoid holes (dark). R = new map."

    clock = pygame.time.Clock()

    while True:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    board, start, goal = generate_board()
                    player = list(start)
                    done = False
                    status = "New map generated! Arrows move. R = regenerate."
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if not done:
                    dx, dy = 0, 0
                    if event.key == pygame.K_LEFT:
                        dx = -1
                    elif event.key == pygame.K_RIGHT:
                        dx = 1
                    elif event.key == pygame.K_UP:
                        dy = -1
                    elif event.key == pygame.K_DOWN:
                        dy = 1

                    if dx != 0 or dy != 0:
                        nx = max(0, min(GRID_W - 1, player[0] + dx))
                        ny = max(0, min(GRID_H - 1, player[1] + dy))
                        player[0], player[1] = nx, ny

                        tile = board[ny][nx]
                        if tile == HOLE:
                            status = "You fell into a hole! Press R to try a new map."
                            done = True
                        elif tile == GOAL:
                            status = "You reached the goal! Press R for a new random map."
                            done = True

        draw_board(screen, board, tuple(player), font, status)


if __name__ == "__main__":
    main()
