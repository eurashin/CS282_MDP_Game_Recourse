import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import copy

num_rows = 5 # encodes taxi's row
num_cols = 5 # encodes taxi's column
num_passenger_locations = 5 # where passenger is located
num_destinations = 4 # where destination location

class TaxiQLearner:
  def __init__(self, env, epsilon = 0.1, gamma = 0.5, alpha = 0.6):
    self.env = env
    # Initialize Q-table
    self.num_states = num_rows * num_cols * num_passenger_locations * num_destinations
    self.num_actions = 6 # 6 actions: south, north, east, west, pickup passenger, dropoff passenger
    self.Q = np.zeros((self.num_states, self.num_actions)) # states x actions

    # Initialize learning parametesr
    self.epsilon = epsilon # e-greedy
    self.gamma = gamma # discount rate
    self.alpha = alpha # step size

  def policy(self, state): 
      # E-greedy 
      rand = np.random.uniform(0, 1)
      if np.sum(self.Q[state, :]) == 0 or rand <= self.epsilon: #random
        action = self.env.action_space.sample()
      else: # greedy
        max_actions = np.argwhere(self.Q[state,:] == np.amax(self.Q[state,:])).flatten()
        action = np.random.choice(max_actions)
      return action

  # reset agent
  def reset(self):
    self.Q = np.zeros((self.num_states, self.num_actions)) # states x actions    
  
  def train(self, num_episodes):
    data = []
    rewards = []
    episode_size = []
    for episode in range(num_episodes):
        D, r = self.run_episode()
        self.alpha = max(self.alpha - (1e-2 * self.alpha), 1e-3)
        data.append(D)
        rewards.append(r)
        episode_size.append(D.shape[0])
        # Callback
        if episode % 25 == 0: 
          print("Episode {} latest length {}".format(episode, D.shape[0]))
    return np.vstack(data), np.array(rewards), np.array(episode_size)

  # Runs the TaxiV3 environment for the specified number of episodes WITHOUT changing the agent's Q-table
  # Returns: each of the following array are length n, where n is the number of episodes: 
  #        * total_steps: the number of steps the agent took during each episode
  #        * total_penalties: the number of penalties (illegal moves) agent made durinig the episode
  #        * total_rewards: the number 
  def evaluate(self, num_episodes):
    total_steps = np.zeros(num_episodes)
    total_penalties = np.zeros(num_episodes)
    total_rewards = np.zeros(num_episodes)
    for i in range(num_episodes): # run an episode
      state = self.env.reset() # start a new episode
      done = False
      steps = 0
      penalties = 0
      rewards = 0
      while not done: 
        # Perform action
        action = self.policy(state)
        next_state, reward, done, info = self.env.step(action)
        # Track data
        if reward == -10: 
          penalties += 1
        rewards += reward
        steps += 1
        state = next_state

        total_steps[i] = steps
        total_penalties[i] = penalties
        total_rewards[i] = rewards
      print("Evaluation episode {}: {} steps".format(i, steps))
        
    return total_steps, total_penalties, total_rewards


  # Runs a single episode of TaxiV3. The passenger pickup and dropoff locations are randomized at the beginning of the episode. 
  # If train = True, then the agent's Q-table will be updated based on the results of this episode. 
  # Returns: a numpy array of (state, action, reward, next state) tuples and the cumulative rewards for the episode
  def run_episode(self, train = True): 
    curr_state = self.env.reset() # start a new episode
    curr_action = self.policy(curr_state)

    done = False
    D = [] # data of trajectory
    total_rewards = 0
    while not done: 
      # Perform action
      next_state, reward, done, info = self.env.step(curr_action)
      # Track data
      total_rewards += reward
      experience = list(self.env.decode(curr_state)) # s
      experience.extend([curr_state]) # encoded current state
      experience.extend([curr_action]) # a
      experience.extend([reward]) # r
      experience.extend(list(self.env.decode(next_state))) # s'
      experience.extend([next_state]) # encoded next state
      D.append(experience) 

      # Get next action
      next_action = self.policy(next_state)

      # Update Q-values
      if train: 
        self.Q[curr_state, curr_action] = self.Q[curr_state, curr_action] + self.alpha * ((reward + self.gamma * self.Q[next_state, next_action]) - self.Q[curr_state, curr_action])

      curr_state = copy.deepcopy(next_state)
      curr_action = copy.deepcopy(next_action)


    return np.array(D), total_rewards

  def render(self):
    self.env.render()



# Define an RL environment which finds a Q matrix from user rewards
class UserTaxiEnvironment:
  def __init__(self, user_reward, env, multiplier = 1.):
    self.reward = user_reward
    self.env = env
    self.action_space = env.action_space
    self.multiplier = multiplier

  def reset(self):
    return(self.env.reset())

  def decode(self, state):
    return self.env.decode(state)

  def step(self, action):
    next_state, global_reward, done, info = self.env.step(action)
    user_reward = self.reward[next_state]
    return next_state, global_reward + (self.multiplier * user_reward), done, info
