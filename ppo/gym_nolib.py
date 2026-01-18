import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical

# ========================================
# STEP 1: ENVIRONMENT & HYPERPARAMETERS
# ========================================
problem = 'LunarLander-v3'
env = gym.make(problem, render_mode=None)

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_LOSS_COEF = 0.5
BATCH_SIZE = 64
PPO_EPOCHS = 10
ROLLOUT_STEPS = 2048
TOTAL_STEPS = 500_000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========================================
# STEP 2: SEPARATE ACTOR AND CRITIC NETWORKS
# ========================================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state):
        # Outputs logits for categorical distribution
        return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1) # Output scalar Value
        )
        
    def forward(self, state):
        return self.net(state)

# Setup Networks
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

actor = Actor(state_dim, action_dim).to(DEVICE)
critic = Critic(state_dim).to(DEVICE)

# Combine parameters into one optimizer
optimizer = optim.Adam(
    list(actor.parameters()) + list(critic.parameters()), 
    lr=LEARNING_RATE
)

# Helper to keep usage clean (similar to previous "get_action_and_value")
def get_action_and_value(state, action=None):
    # 1. Actor pass
    logits = actor(state)
    probs = Categorical(logits=logits)
    if action is None:
        action = probs.sample()
    
    # 2. Critic pass (completely separate gradients)
    value = critic(state)
    
    return action, probs.log_prob(action), probs.entropy(), value

# ========================================
# STEP 3: GAE (Unchanged)
# ========================================
def compute_gae(rewards, values, dones, next_value):
    advantages = []
    last_gae_lam = 0
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            next_non_terminal = 1.0 - dones[step]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_val = values[step + 1]
            
        delta = rewards[step] + GAMMA * next_val * next_non_terminal - values[step]
        last_gae_lam = delta + GAMMA * GAE_LAMBDA * next_non_terminal * last_gae_lam
        advantages.insert(0, last_gae_lam)
    return torch.tensor(advantages, dtype=torch.float32).to(DEVICE)

# ========================================
# STEP 4: UPDATE LOGIC
# ========================================
def ppo_update(states, actions, log_probs, returns, advantages):
    dataset_size = states.size(0)
    indices = np.arange(dataset_size)
    
    for _ in range(PPO_EPOCHS):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, BATCH_SIZE):
            end = start + BATCH_SIZE
            idx = indices[start:end]
            
            # CALL THE HELPER (Uses both networks)
            _, new_log_probs, entropy, new_values = get_action_and_value(states[idx], actions[idx])
            
            # Ratio and Surrogate Loss
            log_ratio = new_log_probs - log_probs[idx]
            ratio = log_ratio.exp()
            
            mb_advantages = advantages[idx]
            mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * mb_advantages
            
            # Policy Loss (Actor)
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value Loss (Critic)
            value_loss = 0.5 * ((new_values.squeeze() - returns[idx]) ** 2).mean()
            
            # Total Loss
            loss = policy_loss + VALUE_LOSS_COEF * value_loss - ENTROPY_COEF * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients for both networks
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            
            optimizer.step()

# ========================================
# STEP 5: TRAINING LOOP
# ========================================
def train():
    print("Starting training with Separated Networks...")
    global_step = 0
    state, _ = env.reset()
    
    while global_step < TOTAL_STEPS:
        batch_states, batch_actions, batch_log_probs = [], [], []
        batch_rewards, batch_dones, batch_values = [], [], []
        
        for _ in range(ROLLOUT_STEPS):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
                
                # Helper calls both networks
                action, log_prob, _, value = get_action_and_value(state_t)
            
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            batch_states.append(state_t)
            batch_actions.append(action)
            batch_log_probs.append(log_prob)
            batch_values.append(value.item())
            batch_rewards.append(reward)
            batch_dones.append(done)
            
            state = next_state
            global_step += 1
            
            if done:
                state, _ = env.reset()
        
        # Bootstrapping
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            # Use Critic directly for bootstrap value
            next_value = critic(state_t).squeeze().item()
            
        advantages = compute_gae(batch_rewards, batch_values, batch_dones, next_value)
        returns = advantages + torch.tensor(batch_values).to(DEVICE)
        
        b_states = torch.cat(batch_states)
        b_actions = torch.cat(batch_actions)
        b_log_probs = torch.cat(batch_log_probs)
        
        ppo_update(b_states, b_actions, b_log_probs, returns, advantages)
        
        print(f"Step: {global_step}/{TOTAL_STEPS} | Mean Reward: {np.mean(batch_rewards) * ROLLOUT_STEPS / sum(batch_dones):.2f}")

if __name__ == "__main__":
    train()