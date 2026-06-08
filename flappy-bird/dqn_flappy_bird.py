#!/usr/bin/env python3
"""
DQN agent for Flappy Bird environment using Stable-Baselines3.
"""

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.monitor import Monitor
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import and register the environment
from register_env import *

def create_model(env, seed=42):
    """Create DQN model with appropriate hyperparameters for Flappy Bird."""

    # Set random seed for reproducibility
    th.manual_seed(seed)
    np.random.seed(seed)

    # Create DQN model with tuned hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=32,
        tau=1.0,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        max_grad_norm=10,
        verbose=1,
        seed=seed,
        device="auto"  # Use GPU if available, otherwise CPU
    )

    return model

def train_dqn_agent(total_timesteps=500000, save_path="checkpoints/dqn_flappy_bird"):
    """Train DQN agent on Flappy Bird environment."""

    print("Creating Flappy Bird environment...")

    # Create environment
    env = gym.make('FlappyBird-v0')

    # Wrap environment with Monitor to record stats
    env = Monitor(env)

    print("Creating DQN model...")
    model = create_model(env)

    # Create evaluation environment and callback
    eval_env = gym.make('FlappyBird-v0')
    eval_env = Monitor(eval_env)

    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=callback_on_best,
        verbose=1,
        best_model_save_path=save_path,
        log_path=save_path,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    print(f"Training DQN agent for {total_timesteps} timesteps...")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        log_interval=1000
    )

    # Save the final model
    model.save(f"{save_path}/final_model")
    print(f"Model saved to {save_path}/final_model")

    return model

def test_trained_agent(model_path="checkpoints/dqn_flappy_bird/final_model", episodes=10):
    """Test a trained DQN agent."""

    print("Loading trained model...")

    # Load the trained model
    model = DQN.load(model_path)

    # Create environment with rendering
    env = gym.make('FlappyBird-v0', render_mode='human')

    print(f"Testing trained agent for {episodes} episodes...")

    scores = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        score = 0

        while not done:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            score = info['score']

        scores.append(score)
        print(f"Episode {episode + 1}: Score = {score}, Total Reward = {total_reward}")

    env.close()

    print(f"\nAverage Score: {np.mean(scores):.2f}")
    print(f"Best Score: {np.max(scores)}")

    return scores

def evaluate_agent_performance(model_path="checkpoints/dqn_flappy_bird/final_model", episodes=100):
    """Evaluate agent performance over many episodes without rendering."""

    print("Loading trained model for evaluation...")

    # Load the trained model
    model = DQN.load(model_path)

    # Create environment without rendering for faster evaluation
    env = gym.make('FlappyBird-v0')

    print(f"Evaluating agent performance over {episodes} episodes...")

    scores = []
    rewards = []

    for episode in range(episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        score = 0

        while not done:
            # Predict action using the trained model
            action, _states = model.predict(obs, deterministic=True)

            # Take action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            score = info['score']

        scores.append(score)
        rewards.append(total_reward)

        if (episode + 1) % 20 == 0:
            print(f"Episode {episode + 1}/{episodes}: Current Avg Score = {np.mean(scores):.2f}")

    env.close()

    print("\n=== Evaluation Results ===")
    print(f"Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Median Score: {np.median(scores):.2f}")
    print(f"Best Score: {np.max(scores)}")
    print(f"Worst Score: {np.min(scores)}")
    print(f"Success Rate (score > 0): {100 * np.mean(np.array(scores) > 0):.1f}%")
    print(f"Average Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    return scores, rewards

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DQN Agent for Flappy Bird")
    parser.add_argument("--mode", choices=["train", "test", "evaluate"], default="train",
                        help="Mode: train, test, or evaluate")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Number of training timesteps")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes for testing/evaluation")
    parser.add_argument("--model-path", default="checkpoints/dqn_flappy_bird/final_model",
                        help="Path to trained model")

    args = parser.parse_args()

    if args.mode == "train":
        print("Training DQN agent...")
        model = train_dqn_agent(total_timesteps=args.timesteps)
        print("Training completed!")

    elif args.mode == "test":
        print("Testing trained agent...")
        scores = test_trained_agent(model_path=args.model_path, episodes=args.episodes)
        print("Testing completed!")

    elif args.mode == "evaluate":
        print("Evaluating agent performance...")
        scores, rewards = evaluate_agent_performance(model_path=args.model_path, episodes=args.episodes)
        print("Evaluation completed!")
