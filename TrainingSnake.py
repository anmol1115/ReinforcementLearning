from Environments.Snake import Env
from RLClasses import epsilonByFrame, ReplayBuffer, DQNAgent, DuelingDQNAgent, DQNHelper, DDQNHelper

env = Env()
action_size = env.action_size
obs_size = env.observation_size

rb = ReplayBuffer(Capacity=1000)
agent = DQNAgent(obs_size, action_size)

def dqnTraining():
	helper = DQNHelper(agent, action_size)

	max_frames = 10000
	batch_size = 100
	episode_reward = 0
	total_rewards = []

	state = env.reset()
	for frame in range(max_frames):
		epsilon = epsilonByFrame(frame)
		action = helper.act(epsilon, state)

		next_state, reward, done = env.step(action)
		rb.push(state, action, reward, next_state, done)
		state = next_state
		episode_reward += reward

		if done:
			total_rewards.append(episode_reward)
			episode_reward = 0
			state = env.reset()

		if len(rb) > batch_size:
			helper.compute_loss(batch_size, rb)
		else:
			helper.compute_loss(len(rb), rb)

	helper.plot(total_rewards)

def ddqnTraining():
	target_agent = DQNAgent(obs_size, action_size)
	helper = DDQNHelper(agent, target_agent, action_size)

	max_frames = 10000
	batch_size = 100
	episode_reward = 0
	total_rewards = []

	state = env.reset()
	for frame in range(max_frames):
		epsilon = epsilonByFrame(frame)
		action = helper.act(epsilon, state)

		next_state, reward, done = env.step(action)
		rb.push(state, action, reward, next_state, done)
		state = next_state
		episode_reward += reward

		if done:
			total_rewards.append(episode_reward)
			episode_reward = 0
			state = env.reset()

		if len(rb) > batch_size:
			helper.compute_loss(batch_size, rb)
		else:
			helper.compute_loss(len(rb), rb)

		if frame%100 == 0:
			helper.update_target_network()

	helper.plot(total_rewards)

def duelingdqnTraining():
	agent = DuelingDQNAgent(obs_size, action_size)
	target_agent = DuelingDQNAgent(obs_size, action_size)
	helper = DDQNHelper(agent, target_agent, action_size)

	max_frames = 10000
	batch_size = 100
	episode_reward = 0
	total_rewards = []

	state = env.reset()
	for frame in range(max_frames):
		epsilon = epsilonByFrame(frame)
		action = helper.act(epsilon, state)

		next_state, reward, done = env.step(action)
		rb.push(state, action, reward, next_state, done)
		state = next_state
		episode_reward += reward

		if done:
			total_rewards.append(episode_reward)
			episode_reward = 0
			state = env.reset()

		if len(rb) > batch_size:
			helper.compute_loss(batch_size, rb)
		else:
			helper.compute_loss(len(rb), rb)

		if frame%100 == 0:
			helper.update_target_network()

	helper.plot(total_rewards)

if __name__ == "__main__":
	# dqnTraining()
	# ddqnTraining()
	duelingdqnTraining()