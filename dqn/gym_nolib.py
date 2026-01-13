import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import sys
import collections
import random
from typing import Tuple, NamedTuple

# ========================================
# STEP 1: ENVIRONMENT SETUP
# ========================================
# We use Gymnasium's CartPole-v1 environment.
# - Observation space: Box(4,) -> [cart position, cart velocity, pole angle, pole angular velocity]
# - Action space: Discrete(2) -> 0: move cart left, 1: move cart right
# - Episode terminates if pole angle > ±12°, cart position > ±2.4, or >500 steps (solved threshold).
# Goal: Maximize episode length (balance the pole as long as possible).
env = gym.make('LunarLander-v3', render_mode=None)  # No rendering for training speed
state, info = env.reset()
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")
# Output example:
# Observation space: Box(-inf, inf, (4,), float32)
# Action space: Discrete(2)

# ========================================
# STEP 2: HYPERPARAMETERS
# ========================================
# Key DQN hyperparameters tuned for CartPole (common values that work well).
BUFFER_SIZE = int(1e5)      # Max experiences in replay buffer
BATCH_SIZE = 64             # Mini-batch size for training
GAMMA = 0.99                # Discount factor for future rewards
EPSILON_START = 1.0         # Initial exploration rate
EPSILON_END = 0.01          # Final (minimum) exploration rate
EPSILON_DECAY = 0.995       # Decay rate per episode
LEARNING_RATE = 5e-4        # Adam optimizer learning rate
TARGET_UPDATE_FREQ = 1000   # Steps between target network updates (tau=1 soft update alternative)
NUM_EPISODES = 5000         # Total training episodes
MAX_STEPS_PER_EPISODE = 500 # Cap per episode (env default)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ========================================
# STEP 3: REPLAY BUFFER
# ========================================
# Experience Replay: Store transitions (s, a, r, s', done) to break correlation in sequential data.
# Sample random mini-batches for i.i.d. updates, stabilizing training.
Transition = NamedTuple('Transition', [('state', np.ndarray),
                                       ('action', int),
                                       ('reward', float),
                                       ('next_state', np.ndarray),
                                       ('done', bool)])

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)
    
    def push(self, transition: Transition):
        """Add a transition to the buffer."""
        self.buffer.append(transition)
    
    def sample(self, batch_size: int) -> Tuple:
        """Randomly sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        # Stack tensors for batch processing
        state_batch = torch.FloatTensor([t.state for t in batch]).to(DEVICE)
        action_batch = torch.LongTensor([t.action for t in batch]).to(DEVICE).unsqueeze(1)
        reward_batch = torch.FloatTensor([t.reward for t in batch]).to(DEVICE)
        next_state_batch = torch.FloatTensor([t.next_state for t in batch]).to(DEVICE)
        done_batch = torch.BoolTensor([t.done for t in batch]).to(DEVICE)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch
    
    def __len__(self) -> int:
        return len(self.buffer)

replay_buffer = ReplayBuffer(BUFFER_SIZE)

# ========================================
# STEP 4: Q-NETWORK (NEURAL NETWORK)
# ========================================
# Deep Q-Network: Approximates Q(s, a) = expected discounted future reward from state s taking action a.
# Architecture: Simple MLP for low-dim state (4 inputs, 2 outputs).
# - Input: state (4,)
# - Hidden: 128 -> 128 (ReLU)
# - Output: Q-values for each action (2,)
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Raw Q-values (no softmax; use argmax)

state_dim = env.observation_space.shape[0]  
action_dim = env.action_space.n            
print(f"state_dim: {state_dim}, action_dim: {action_dim}")
q_network = QNetwork(state_dim, action_dim).to(DEVICE)
target_network = QNetwork(state_dim, action_dim).to(DEVICE)  # Target net starts as copy of q_net
target_network.load_state_dict(q_network.state_dict())      # Hard copy

# ========================================
# STEP 5: OPTIMIZER AND LOSS
# ========================================
optimizer = optim.Adam(q_network.parameters(), lr=LEARNING_RATE)
# Loss: MSE between predicted Q(s,a) and target Q(s,a)

# ========================================
# STEP 6: EPSILON-GREEDY POLICY
# ========================================
# Epsilon-greedy: With prob epsilon, random action (explore); else argmax Q(s,a) (exploit).
# Epsilon decays over episodes for more exploitation later.

# ========================================
# STEP 7: TRAINING LOOP
# ========================================
# Main loop: Episodes -> Steps -> Interact, Store, Sample, Update.
def train():
    epsilon = EPSILON_START
    episode_rewards = []
    step_count = 0  # Global step counter for target updates
    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(MAX_STEPS_PER_EPISODE):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()  # Random (explore)
            else:
                with torch.no_grad():
                    q_values = q_network(torch.FloatTensor(state).to(DEVICE))
                    action = q_values.argmax().item()  # Greedy (exploit)
            
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            
            # Store transition (normalize state? Optional for CartPole)
            replay_buffer.push(Transition(state, action, reward, next_state, done))
            
            # Train if enough samples
            if len(replay_buffer) >= BATCH_SIZE:
                # Sample mini-batch
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = replay_buffer.sample(BATCH_SIZE)
                # Q(s,a) prediction from q_network
                q_values = q_network(state_batch)
                q_values = q_values.gather(1, action_batch)
                # Target Q(s', a') using target_network (double DQN: argmax from q_net, value from target)
                with torch.no_grad():
                    # Double DQN: select action with q_network, evaluate with target_network
                    best_actions = q_network(next_state_batch).argmax(dim=1, keepdim=True)
                    next_q = target_network(next_state_batch).gather(1, best_actions)
                    target_q = reward_batch.unsqueeze(1) + (~done_batch).unsqueeze(1).float() * GAMMA * next_q
                # Loss: MSE (Bellman error)
                loss = F.mse_loss(q_values, target_q)
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                # Clip gradients for stability
                torch.nn.utils.clip_grad_value_(q_network.parameters(), 1.0)
                optimizer.step()
            
            state = next_state
            step_count += 1
            
            if done:
                break
        
        # Decay epsilon
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        
        # Update target network every TARGET_UPDATE_FREQ steps (hard update)
        if step_count % TARGET_UPDATE_FREQ == 0:
            target_network.load_state_dict(q_network.state_dict())
        
        episode_rewards.append(episode_reward)
        
        # Logging (every 100 episodes)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode+1}/{NUM_EPISODES}, Avg Reward (last 100): {avg_reward:.2f}, "
                f"Epsilon: {epsilon:.3f}, Buffer: {len(replay_buffer)}")
        
        torch.save(q_network.state_dict(), 'cartpole_dqn.pth')

### ========================================
# STEP 8: EVALUATION
# ========================================
# Test trained agent with epsilon=0 (pure greedy).
def evaluate(num_episodes: int = 1000):
    checkpoint = torch.load('cartpole_dqn.pth', weights_only=False)
    q_network.load_state_dict(checkpoint)
    q_network.eval()
    env = gym.make('LunarLander-v3', render_mode='human')  # Render for viz
    scores = []
    for i in range(num_episodes):
        print(f"Evaluating episode {i+1}/{num_episodes}")
        state, _ = env.reset()
        total_reward = 0
        done = False
        while not done:
            with torch.no_grad():
                q_values = q_network(torch.FloatTensor(state).to(DEVICE))
                action = q_values.argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        scores.append(total_reward)
    env.close()
    print(f"Evaluation Avg Reward: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    return scores

#train()
evaluate()

# ========================================
# DETAILED EXPLANATIONS
# ========================================
# 1. **Environment**: Gymnasium handles physics/stepping. Reset for new episodes.
#
# 2. **Replay Buffer**: FIFO deque for fixed capacity. Random sampling ~ i.i.d. data.
#
# 3. **Q-Network**: MLP approximates Q-function. Forward pass gives Q(s) vector.
#    - Double DQN: Use online net for action selection, target for evaluation (reduces overestimation).
#
# 4. **Target Network**: Frozen copy; slows changing targets, improves stability.
#    - Hard update every C steps (alternative: soft tau-update).
#
# 5. **Bellman Target**: Q_target(s,a) = r + γ max_a' Q_target(s', a')  (if not done)
#    - ~done_batch masks terminal states (no future reward).
#
# 6. **Epsilon Decay**: Linear/multiplicative decay shifts explore -> exploit.
#
# 7. **Training Dynamics**:
#    - Early: High epsilon, buffer fills with random data.
#    - Mid: Q-learning converges; target updates stabilize.
#    - Late: Low epsilon, solves if avg >195/100 episodes.
#
# 8. **Improvements Used**:
#    - Experience Replay + Target Net (vanilla DQN).
#    - Double DQN.
#    - Gradient clipping.
#    - Adam optimizer.
#
# 9. **Expected Results**: Solves CartPole (~200 reward avg) in <1000 episodes.
#    - Monitor avg_reward; >195 = solved.
#
# 10. **Extensions**:
#     - Prioritized Replay (PER).
#     - Dueling DQN.
#     - Noisy Nets.
#     - Frame stacking for partial obs.
#
# Run this script: Should train and solve CartPole reliably!
