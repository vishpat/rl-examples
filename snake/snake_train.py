import os
import argparse
import gymnasium as gym
import game  # Imports and registers 'Snake-v0'
import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomGridExtractor(BaseFeaturesExtractor):
    """
    A high-resolution Convolutional Neural Network (CNN) features extractor that downsamples
    gradually and dynamically based on the input grid dimensions. By avoiding aggressive
    pooling, this architecture preserves exact pixel-level spatial details (e.g. wall distances,
    adjacent body blocks) required for high-scoring plays on all board scales (from 240 to 720+).
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super(CustomGridExtractor, self).__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        grid_h, grid_w = observation_space.shape[1], observation_space.shape[2]
        
        # Determine stride dynamically to maintain pixel-level resolution
        # Small grids use full-resolution stride 1. Medium/large grids downsample by a factor of 2.
        stride = 2 if max(grid_h, grid_w) > 16 else 1
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        
        # Dynamically compute the flattened features size
        with th.no_grad():
            sample_tensor = th.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample_tensor).shape[1]
            
        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

def train(timesteps=100000, seed=42, tb_log=True, obs_type="features", w=640, h=480, reward_shaping=True):
    print(f"Initializing training environment with obs_type='{obs_type}', size={w}x{h}, reward_shaping={reward_shaping}...")
    # Wrap with Monitor for SB3 logging compatibility
    env = gym.make('Snake-v0', render_mode=None, obs_type=obs_type, w=w, h=h, reward_shaping=reward_shaping)
    env = Monitor(env)
    
    # Check if tensorboard is installed if logging is requested
    tb_log_dir = "./tensorboard_logs/" if tb_log else None
    if tb_log_dir:
        try:
            import tensorboard
        except ImportError:
            print("Warning: 'tensorboard' package is not installed. Disabling Tensorboard logging.")
            tb_log_dir = None
            
    # Configure checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=max(5000, timesteps // 5),
        save_path="./checkpoints/",
        name_prefix="dqn_snake"
    )
    
    # Choose appropriate policy
    # Upgraded grid mode uses CnnPolicy to preserve spatial structure of the multi-channel grid
    policy = "MlpPolicy"
    if obs_type in ["grid", "rgb"]:
        policy = "CnnPolicy"
        
    # Custom policy network architecture (MLP / CNN)
    if policy == "MlpPolicy":
        policy_kwargs = dict(net_arch=[128, 128])
    else:
        # Use our custom grid CNN extractor for grid-based state representations
        if obs_type == "grid":
            policy_kwargs = dict(
                features_extractor_class=CustomGridExtractor,
                features_extractor_kwargs=dict(features_dim=128),
            )
        else:
            # For RGB images (raw uint8, 480x640), default NatureCNN is suitable and will normalize automatically
            policy_kwargs = None
    
    print(f"Creating DQN model using policy='{policy}'...")
    model = DQN(
        policy=policy,
        env=env,
        learning_rate=1e-3,
        buffer_size=50000,
        learning_starts=1000,
        batch_size=64,
        tau=1.0,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.15,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.01,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tb_log_dir,
        verbose=1,
        seed=seed
    )
    
    print(f"Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    
    print("Training finished! Saving final model to dqn_snake.zip...")
    model.save("dqn_snake")
    
    env.close()

def evaluate(episodes=5, render=False, obs_type="features", w=640, h=480):
    print(f"\nEvaluating trained agent for {episodes} episodes using obs_type='{obs_type}', size={w}x{h}...")
    
    model_path = "dqn_snake.zip"
    if not os.path.exists(model_path):
        print(f"Error: {model_path} model not found!")
        return
        
    try:
        # Load model and dynamically supply our CustomGridExtractor in custom_objects if needed
        model = DQN.load(model_path, custom_objects={"CustomGridExtractor": CustomGridExtractor})
        
        # Create evaluation env (Disable reward shaping during evaluation to measure pure performance)
        render_mode = "human" if render else None
        env = gym.make('Snake-v0', render_mode=render_mode, obs_type=obs_type, w=w, h=h, reward_shaping=False)
        
        scores = []
        for episode in range(1, episodes + 1):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            
            while not (done or truncated):
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = env.step(action)
                episode_reward += reward
                
            scores.append(info['score'])
            print(f"Episode {episode}: Score = {info['score']}, Total Steps = {info['steps']}")
            
        print(f"Average Score over {episodes} episodes: {sum(scores)/len(scores):.2f}")
        env.close()
        
    except ValueError as e:
        print(f"\n[ValueError] Shape Mismatch Error during evaluation: {e}")
        print("\nPossible Reason: The saved model 'dqn_snake.zip' in your workspace was trained using a different configuration")
        print(f"(different --obs-type, --width, or --height) than what you are currently evaluating.")
        print("\nTo fix this:")
        print("1. Train a new model matching this configuration by running without --eval-only.")
        print("2. Or delete the existing 'dqn_snake.zip' file to start fresh.")
        print("3. Or run the script with the exact parameters that were used for training.")
        return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train and evaluate DQN agent on custom Snake environment.")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on an existing model.")
    parser.add_argument("--obs-type", type=str, default="features", choices=["features", "grid", "rgb"], 
                        help="Observation space type: 'features' (11-element vector), 'grid' (3D state), 'rgb' (pixels).")
    parser.add_argument("--timesteps", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--eval-episodes", type=int, default=5, help="Number of episodes for evaluation.")
    parser.add_argument("--width", type=int, default=640, help="Screen width in pixels (must be multiple of 20).")
    parser.add_argument("--height", type=int, default=480, help="Screen height in pixels (must be multiple of 20).")
    parser.add_argument("--no-shaping", action="store_true", help="Disable distance-based reward shaping.")
    parser.add_argument("--render", action="store_true", help="Render evaluation gameplay (uses Pygame GUI).")
    parser.add_argument("--no-tb", action="store_true", help="Disable Tensorboard logging.")
    
    args = parser.parse_args()
    
    # Validation checks for board dimensions
    if args.width % 20 != 0 or args.height % 20 != 0:
        raise ValueError("Error: Width and height must be positive integers that are multiples of 20 (BLOCK_SIZE).")
        
    if args.eval_only:
        evaluate(episodes=args.eval_episodes, render=args.render, obs_type=args.obs_type, w=args.width, h=args.height)
    else:
        # Run training, then evaluate
        train(
            timesteps=args.timesteps,
            tb_log=not args.no_tb,
            obs_type=args.obs_type,
            w=args.width,
            h=args.height,
            reward_shaping=not args.no_shaping
        )
        evaluate(episodes=args.eval_episodes, render=args.render, obs_type=args.obs_type, w=args.width, h=args.height)
