import pygame
import sys
import random

# Initialize pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
GRAVITY = 0.25
FLAP_POWER = -5
PIPE_SPEED = 3
PIPE_GAP = 150
PIPE_FREQUENCY = 1800  # milliseconds
GROUND_HEIGHT = 100
FONT_SIZE = 32

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 128, 0)
BLUE = (0, 191, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)

class Bird:
    def __init__(self):
        self.x = 100
        self.y = SCREEN_HEIGHT // 2
        self.velocity = 0
        self.radius = 20

    def flap(self):
        self.velocity = FLAP_POWER

    def update(self):
        self.velocity += GRAVITY
        self.y += self.velocity

    def draw(self, screen):
        pygame.draw.circle(screen, YELLOW, (self.x, self.y), self.radius)
        # Draw eye
        pygame.draw.circle(screen, BLACK, (self.x + 10, self.y - 5), 5)

    def get_rect(self):
        return pygame.Rect(self.x - self.radius, self.y - self.radius,
                          self.radius * 2, self.radius * 2)

class Pipe:
    def __init__(self):
        self.x = SCREEN_WIDTH
        self.height = random.randint(150, SCREEN_HEIGHT - GROUND_HEIGHT - PIPE_GAP - 50)
        self.passed = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, screen):
        # Draw top pipe
        pygame.draw.rect(screen, GREEN, (self.x, 0, 70, self.height))
        # Draw bottom pipe
        bottom_pipe_y = self.height + PIPE_GAP
        bottom_pipe_height = SCREEN_HEIGHT - bottom_pipe_y - GROUND_HEIGHT
        pygame.draw.rect(screen, GREEN, (self.x, bottom_pipe_y, 70, bottom_pipe_height))

    def collide(self, bird):
        bird_rect = bird.get_rect()
        # Top pipe rect
        top_pipe_rect = pygame.Rect(self.x, 0, 70, self.height)
        # Bottom pipe rect
        bottom_pipe_rect = pygame.Rect(self.x, self.height + PIPE_GAP, 70,
                                      SCREEN_HEIGHT - self.height - PIPE_GAP - GROUND_HEIGHT)

        return bird_rect.colliderect(top_pipe_rect) or bird_rect.colliderect(bottom_pipe_rect)

    def off_screen(self):
        return self.x < -70

class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, FONT_SIZE)

        self.reset()

    def reset(self):
        self.bird = Bird()
        self.pipes = []
        self.score = 0
        self.game_over = False
        self.last_pipe = pygame.time.get_ticks()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if self.game_over:
                        self.reset()
                    else:
                        self.bird.flap()
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

    def update(self):
        if not self.game_over:
            # Update bird
            self.bird.update()

            # Check if bird hit the ground or ceiling
            if self.bird.y > SCREEN_HEIGHT - GROUND_HEIGHT - self.bird.radius or self.bird.y < self.bird.radius:
                self.game_over = True

            # Generate new pipes
            current_time = pygame.time.get_ticks()
            if current_time - self.last_pipe > PIPE_FREQUENCY:
                self.pipes.append(Pipe())
                self.last_pipe = current_time

            # Update pipes and check for collisions
            for pipe in self.pipes[:]:
                pipe.update()

                # Check if bird passed the pipe
                if not pipe.passed and pipe.x < self.bird.x:
                    pipe.passed = True
                    self.score += 1

                # Check for collision
                if pipe.collide(self.bird):
                    self.game_over = True

                # Remove pipes that are off screen
                if pipe.off_screen():
                    self.pipes.remove(pipe)

    def draw(self):
        # Draw sky background
        self.screen.fill(BLUE)

        # Draw pipes
        for pipe in self.pipes:
            pipe.draw(self.screen)

        # Draw ground
        pygame.draw.rect(self.screen, GREEN, (0, SCREEN_HEIGHT - GROUND_HEIGHT, SCREEN_WIDTH, GROUND_HEIGHT))

        # Draw bird
        self.bird.draw(self.screen)

        # Draw score
        score_text = self.font.render(str(self.score), True, WHITE)
        self.screen.blit(score_text, (SCREEN_WIDTH // 2 - score_text.get_width() // 2, 50))

        # Draw game over message
        if self.game_over:
            game_over_text = self.font.render("Game Over!", True, RED)
            restart_text = self.font.render("Press SPACE to restart", True, WHITE)
            self.screen.blit(game_over_text,
                           (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - 50))
            self.screen.blit(restart_text,
                           (SCREEN_WIDTH // 2 - restart_text.get_width() // 2, SCREEN_HEIGHT // 2))

        pygame.display.flip()

    def run(self):
        while True:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(60)

if __name__ == "__main__":
    game = Game()
    game.run()