import gym
from gym import spaces
import numpy as np
from gym.envs.registration import register

register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)


env = gym.make('FrozenLakeNotSlippery-v0')

# Initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n]) #16x4 matrix of zeros

# Set learning parameters
lr = .8
y = .95
num_episodes = 2000

#create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
	# Reset environment and get first new observation
	s = env.reset()
	rAll = 0
	d = False
	j = 0
	# The Q-Table learning algorithm
	while j < 99:
		j+=1
		# Choose an action by greedily (with noise) picking from Q table
		# (0 = left, 1 = down, 2=right, 3=up)
		# Less and less random over time (num_episodes)
		a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))

		# Get new state and reward from environment
		# 		observation, reward, done, info = env.step(action)
		s1,r,d,_ = env.step(a)

		# Update Q-Table with new knowledge
		Q[s,a] = Q[s,a] + lr*(r+y*np.max(Q[s1,:]) - Q[s,a])
		rAll += r
		s = s1
		if d == True:
			break
	# print(i, Q)
	# input()
	rList.append(rAll)

print("Score over time: " + str(sum(rList)/num_episodes))

print("Final Q-Table Values:")
print(Q)