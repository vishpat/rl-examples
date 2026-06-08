#!/usr/bin/env python3
"""
Script to register the Flappy Bird environment with Gymnasium.
This allows using gym.make('FlappyBird-v0') to create the environment.
"""

import gymnasium as gym
from gymnasium.envs.registration import register
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment
from flappy_bird_env import FlappyBirdEnv

# Register the environment
register(
    id='FlappyBird-v0',
    entry_point='flappy_bird_env:FlappyBirdEnv',
    max_episode_steps=10000,
)

print("FlappyBird-v0 environment registered successfully!")
print("You can now create it using: gym.make('FlappyBird-v0')")