import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('CartPole-v1')
observation, info = env.reset()
print(f"Observation: {observation}")
print(f"Info: {info}")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

print(state_size, action_size)

action = env.action_space.sample()
print(f"Action: {action}")
print(f"Observation: {env.step(action)}")

env.close()

class DQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 24)
        self.fc2 = nn.Linear(24, action_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x

model = DQN(state_size, action_size)
print(model)