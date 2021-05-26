import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f

import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

def epsilonByFrame(index, decay_rate=500):
	epsilon_start = 1
	epsilon_end = 0.01
	epsilon_decay = decay_rate

	return epsilon_end + (epsilon_start - epsilon_end)*np.exp(-1 * index / epsilon_decay)

class ReplayBuffer:
	def __init__(self, Capacity=1000):
		self.memory = deque(maxlen=Capacity)

	def push(self, state, action, reward, next_state, done):
		state = np.expand_dims(state, 0)
		next_state = np.expand_dims(next_state, 0)
		self.memory.append((state, action, reward, next_state, done))

	def sample(self, batch_size):
		states, actions, rewards, next_states, dones = zip(*random.sample(self.memory, batch_size))
		return np.concatenate(states), actions, rewards, np.concatenate(next_states), dones

	def __len__(self):
		return len(self.memory)

class DQNAgent(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.fc1 = nn.Linear(in_features=input_size, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=output_size)

	def forward(self, t):
		t = f.relu(self.fc1(t))
		t = self.fc2(t)
		return t

class DuelingDQNAgent(nn.Module):
	def __init__(self, input_size, output_size):
		super().__init__()
		self.fc1 = nn.Linear(in_features=input_size, out_features=256)
		self.fc2 = nn.Linear(in_features=256, out_features=64)
		self.fc3 = nn.Linear(in_features=64, out_features=output_size)
		self.fc4 = nn.Linear(in_features=256, out_features=64)
		self.fc5 = nn.Linear(in_features=64, out_features=1)

	def forward(self, t):
		t = f.relu(self.fc1(t))

		advantage = f.relu(self.fc2(t))
		advantage = self.fc3(advantage)

		value = f.relu(self.fc4(t))
		value = self.fc5(value)

		return value + advantage - advantage.mean()

class DQNHelper:
	def __init__(self, model, action_size):
		self.gamma = 0.9
		self.model = model
		self.action_size = action_size
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

	def act(self, epsilon, state):
		if random.random() > epsilon:
			state0 = torch.tensor(state, dtype=torch.float32)
			action = torch.argmax(self.model(state0)).item()
		else:
			action = random.randint(0, self.action_size-1)
		return action

	def compute_loss(self, batch_size, memory):
		state, action, reward, next_state, done = memory.sample(batch_size)
		state = torch.tensor(state, dtype=torch.float32)
		action = torch.tensor(action, dtype=torch.int64)
		reward = torch.tensor(reward, dtype=torch.float32)
		next_state = torch.tensor(next_state, dtype=torch.float32)
		done = torch.tensor(done, dtype=torch.float32)

		q_values = self.model(state)
		next_q_values = self.model(next_state)

		q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
		next_q_value = next_q_values.max(1)[0]
		expected_q_value = reward + self.gamma*next_q_value*(1-done)

		loss = (q_value - expected_q_value).pow(2).mean()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def plot(self, arr):
		plt.plot(arr)
		avg_list = []
		for element in range(1, len(arr)+1):
			avg_list.append(sum(arr[:element])/len(arr[:element]))
		plt.plot(avg_list)
		plt.show()

class DDQNHelper(DQNHelper):
	def __init__(self, model, target_model, action_size):
		super().__init__(model, action_size)
		self.target_model = target_model

	def compute_loss(self, batch_size, memory):
		state, action, reward, next_state, done = memory.sample(batch_size)
		state = torch.tensor(state, dtype=torch.float32)
		action = torch.tensor(action, dtype=torch.int64)
		reward = torch.tensor(reward, dtype=torch.float32)
		next_state = torch.tensor(next_state, dtype=torch.float32)
		done = torch.tensor(done, dtype=torch.float32)

		q_values = self.model(state)
		next_q_values = self.model(next_state)
		next_q_state_values = self.target_model(next_state)

		q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
		next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)
		expected_q_value = reward + self.gamma*next_q_value*(1-done)

		loss = (q_value - expected_q_value).pow(2).mean()
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()

	def update_target_network(self):
		self.target_model.load_state_dict(self.model.state_dict())