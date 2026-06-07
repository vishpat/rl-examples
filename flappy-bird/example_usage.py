#!/usr/bin/env python3
"""
Example usage of the Flappy Bird environment.
"""

import gymnasium as gym
import numpy as np
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the environment
from flappy_bird_env import FlappyBirdEnv

def random_agent_demo(episodes=3):
    """Demonstrate using the environment with a random agent."""
    print("Flappy Bird Environment Demo")
    print("=" * 30)

    # Create environment
    env = FlappyBirdEnv()

    for episode in range(episodes):
        print(f"\nEpisode {episode + 1}")
        print("-" * 20)

        # Reset the environment
        observation, info = env.reset()
        print(f"Initial observation: {observation}")

        total_reward = 0
        steps = 0

        # Run episode
        done = False
        while not done:
            # Choose random action
            action = env.action_space.sample()

            # Take action
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            steps += 1

            # Print periodic updates
            if steps % 50 == 0 or done:
                print(f"Step {steps}: Score = {info['score']}, Reward = {reward}")

        print(f"Episode finished after {steps} steps")
        print(f"Final score: {info['score']}")
        print(f"Total reward: {total_reward}")

    env.close()
    print("\nDemo completed!")

def observation_analysis(episodes=5):
    """Analyze the observation space."""
    print("\n\nObservation Space Analysis")
    print("=" * 30)

    env = FlappyBirdEnv()

    # Collect observations
    observations = []
    for episode in range(episodes):
        observation, info = env.reset()
        observations.append(observation.copy())

        done = False
        while not done:
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            observations.append(observation.copy())

    env.close()

    # Convert to numpy array for analysis
    obs_array = np.array(observations)

    print(f"Collected {len(observations)} observations")
    print(f"Observation shape: {obs_array.shape}")
    print("\nObservation Statistics:")
    print(f"  Bird Y Position (norm): Min={obs_array[:, 0].min():.3f}, Max={obs_array[:, 0].max():.3f}, Mean={obs_array[:, 0].mean():.3f}")
    print(f"  Bird Velocity:          Min={obs_array[:, 1].min():.3f}, Max={obs_array[:, 1].max():.3f}, Mean={obs_array[:, 1].mean():.3f}")
    print(f"  Distance to Pipe:       Min={obs_array[:, 2].min():.3f}, Max={obs_array[:, 2].max():.3f}, Mean={obs_array[:, 2].mean():.3f}")
    print(f"  Top Pipe Gap:           Min={obs_array[:, 3].min():.3f}, Max={obs_array[:, 3].max():.3f}, Mean={obs_array[:, 3].mean():.3f}")
    print(f"  Bottom Pipe Gap:        Min={obs_array[:, 4].min():.3f}, Max={obs_array[:, 4].max():.3f}, Mean={obs_array[:, 4].mean():.3f}")

if __name__ == "__main__":
    random_agent_demo()
    observation_analysis()