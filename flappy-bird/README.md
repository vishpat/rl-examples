# Flappy Bird Game and RL Environment

This project contains both a playable Flappy Bird game implementation using Pygame and a custom OpenAI Gymnasium environment for reinforcement learning.

## Project Structure

- `flappy_bird.py`: Standalone Flappy Bird game using Pygame
- `flappy_bird_env.py`: Custom Gymnasium environment implementation
- `test_flappy_env.py`: Script to test the Gymnasium environment
- `train_simple_agent.py`: Example of training a Q-learning agent

## Installation

1. Create a virtual environment:
   ```
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## How to Play the Game

1. Run the game:
   ```
   python flappy_bird.py
   ```

2. Controls:
   - Press SPACE to make the bird flap upward
   - Press ESC to quit the game
   - After game over, press SPACE to restart

## Game Rules

- The bird continuously falls due to gravity
- Navigate through the gaps between green pipes
- Each pipe you pass increases your score by 1
- The game ends if the bird collides with a pipe or the ground
- Try to achieve the highest score possible!

## Using the Gymnasium Environment

The custom Flappy Bird environment can be used for reinforcement learning experiments.

### Environment Details

**Observation Space** (5-dimensional vector):
1. Bird's vertical position (normalized to [0, 1])
2. Bird's vertical velocity
3. Horizontal distance to the next pipe
4. Vertical position of the top pipe gap (normalized)
5. Vertical position of the bottom pipe gap (normalized)

**Action Space** (Discrete):
- 0: Do nothing
- 1: Flap (jump)

**Reward Function**:
- +1 for passing through a pipe
- -1 for dying (collision with pipe or ground)
- 0 for each frame survived

### Testing the Environment

Run the test script to verify the environment works:
```
python test_flappy_env.py
```

### Training a Simple Agent

Train a Q-learning agent on the environment:
```
python train_simple_agent.py
```

Note: This will create a plot of training results and save it as `training_results.png`.

## Environment Registration

The environment is registered with ID `FlappyBird-v0` and can be created using:
```python
import gymnasium as gym
env = gym.make('FlappyBird-v0')
```

For visualization during training/testing:
```python
env = gym.make('FlappyBird-v0', render_mode='human')
```