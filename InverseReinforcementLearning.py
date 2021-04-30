import warnings
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import copy

class max_entropy:
  def __init__(self, trajectories, epochs, learning_rate, epsilon, environment, minibatch_size = 100, 
               state_features = ['taxi_row', 'taxi_column', 'passenger_loc', 'destination']):
    # agent trajectory and state feature names
    self.trajectories = trajectories
    self.state_features = state_features
    # open ai environment, so later we can encode and decode state and have access to transition probabilities
    self.env = environment 
    # gradient descent parameters
    self.epochs = epochs 
    self.learning_rate = learning_rate
    self.gd_threshold = epsilon # stopping criterion
    self.verbose = True 
    self.minibatch_size = minibatch_size

    # get terminal states for the taxi open ai
    self.terminal_state = set()
    for trajectory in self.trajectories:
      # the last state where passenger location = destination location
      self.terminal_state.add(trajectory.loc[trajectory.shape[0] - 1, 'new_state']) 

    # Encode trajectories
    self.phi_trajectories = []
    for trajectory in self.trajectories: 
      phi_traj = self.encode_trajectory(trajectory)
      self.phi_trajectories.append(phi_traj)

    # Get transition matrix
    self.P = np.zeros((self.env.nS, self.env.nA, self.env.nS))
    for state in range(self.env.nS):
      for action in range(self.env.nA):
        next_state = self.env.P[state][action][0][1]
        self.P[state, action, next_state] = 1

  # part 1 of the gradient: empirical expected feature counts from M trajectories
  '''
  def expected_feature_count(self):
    
      #return: a (4, ) vector for expected feature count ['taxi_row', 'taxi_column', 'passenger_loc', 'destination']
    
    efc = np.zeros(len(self.state_features)) # initialize an array

    for trajectory in self.trajectories: # for each trajectory
      # calculate sum for each feature
      for i in range(len(self.state_features)):
        feature_name = 'prev_'+ self.state_features[i] # get feature name in the data frame
        efc[i] += trajectory.loc[:, feature_name].sum()
    # take an average over # of trajectories
    return efc / len(self.trajectories)
  '''

  def expected_feature_count(self):
    '''
      return: a (4, ) vector for expected feature count ['taxi_row', 'taxi_column', 'passenger_loc', 'destination']
    '''
    efc = np.zeros((500, 1)) # initialize an array
    traj_i = np.random.choice(range(len(self.trajectories)), self.minibatch_size) # minibatch of trajectory
    batch_trajectories = []
    for i in traj_i: # for each trajectory
      trajectory = self.phi_trajectories[i]
      batch_trajectories.append(self.trajectories[i])
      efc += trajectory

    # take an average over # of trajectories
    return efc / self.minibatch_size, batch_trajectories

  # get probabilities for each state being the initial state
  def s0_probabilities(self):
    '''
      return p_s0 as a (500, ) array
    '''
    p_s0 = np.zeros(self.env.nS)
    # accumulate over initial state s0 for each trajectory
    for trajectory in self.trajectories:
      s0 = trajectory.loc[0, 'prev_state'] # initial state = prev-state in the first row
      # count # times each state being s0
      p_s0[s0] += 1.0
    return p_s0 / len(self.trajectories) # average by # of trajectories -> sum to be 1

  '''
  def deal_with_overflow(self, input):
    if not np.isfinite(input).all():
      finite_values = np.where(np.isfinite(input), input, 1e-6)
      max_value, min_value = finite_values.max(), finite_values.min()
      
      new_input = np.where(np.isposinf(input), max_value, input)
      new_input = np.where(np.isneginf(new_input), min_value, new_input)
      new_input = np.where(np.isnan(new_input), min_value, new_input)  
      return new_input
    else:
      return input
    '''

  def expected_state_visitation_frequency(self, user_reward, batch_trajectories):
    '''
      input:
        user_reward: per-state reward we learn during the optimization
      return p_state_frequency (num_state(500), ) array
    '''
    num_state = self.env.nS
    num_actions = self.env.nA
    print("Backward pass...")
    # backward pass
    # initialize the state partition function at terminal states
    state_partition = np.zeros(num_state, dtype=np.float64)
    state_partition[list(self.terminal_state)] = 1

    # recursively learn the state action and state partition functions
    for i in range(100): # Usually 2 * number of states 
        # initialize the state action partition function
      state_action_partition = np.zeros((num_state, num_actions), dtype=np.float64)

      # for each state action pair, update the parition function
      for prev_state in range(num_state):
        for action in range(num_actions):
          # since we have deterministic MDP, there is only one unique new_state
          new_state = self.env.P[prev_state][action][0][1]
          reward = user_reward[prev_state]
          # avoid overflow: exp(a - b) = exp(a) / exp(b)
          max_reward = user_reward.max()
          if max_reward > 1: 
            state_action_partition[prev_state, action] = 1.0 * reward/max_reward * state_partition[new_state]
          else: 
            state_action_partition[prev_state, action] = 1.0 * reward * state_partition[new_state]
    
      # update the state partition function
      state_partition = state_action_partition.sum(axis = 1) # over different actions


    # get local action probability
    # p_action = state_action_partition / corresponding state_partition
    p_action = state_action_partition / (state_partition.reshape((-1, 1)) + 1e-5)
    #p_action = self.deal_with_overflow(p_action)

    # forward pass
    # initialize D_s_t with d0 probabilities
    T = max([t.shape[0] for t in batch_trajectories])
    D = np.zeros((num_state, T)) # get T = max length of trajectories
    D[:, 0] = self.s0_probabilities()

    print("Forward pass...")
    # iterate over T steps
    for t in range(1, T):
      if t % 100 == 0: 
        print("{}/{}".format(t, T))
      # Update D_s, t
      for s_to in range(num_state):
        # Get all states that could lead to current
        transition_to = self.P[:, :, s_to]
        state_action_froms = np.where(transition_to == 1) # states, actions that lead to s_to
        s_from_values = 0
        for i in range(state_action_froms[0].shape[0]):
          s_from_values += D[state_action_froms[0][i], t-1] * p_action[state_action_froms[0][i], state_action_froms[1][i]]
        
        D[s_to, t] += np.sum(s_from_values)
    
    # sum over different t
    D_return = D.sum(axis = 1).reshape(-1, 1)
    print("Done: ", D_return.shape)
    #D_return = self.deal_with_overflow(D_return)
    return D_return

  def encode_trajectory(self, trajectory):
    # returns one-hot encoded trajectory
    phi_traj = np.zeros((500, 1))
    states, counts = np.unique(trajectory.loc[:, 'prev_state'], return_counts=True) 
    phi_traj[states] = counts.reshape(-1, 1)
    phi_traj[trajectory.iloc[-1, -1]] += 1
    return phi_traj


  # maximum entropy IRL
  def max_entropy_irl(self, feature_map):
    '''
    input:
      feature map: (num_state x num_features) array to get features for each state
    '''
    # initialize reward weight
    reward_weight = np.random.uniform(size = (feature_map.shape[1],1))

    # compute part 1 of the gradient: empirical expected feature count
    # gradient descent optimization
    grads = [] # plot gradients
    reward_diff = self.gd_threshold + 1 # keep track of the difference between current reward and the last reward

    # stop when reach num of epochs or changes in reward < threshold
    for i in range(self.epochs):
      print("Iteration {}".format(i))
      f_delta, batch_trajectories = self.expected_feature_count()
      print(f_delta[:10, 0])

      # per-state reward
      user_reward = feature_map.dot(reward_weight) 
      #print("reward: ",np.max(user_reward))
      # check for reward convergence
      if i > 0:
        reward_diff = np.max(abs(user_reward - last_reward))
      if reward_diff <= self.gd_threshold and i > 0:
        break # reward converge
      else:
        # compute gradient
        state_visitation_frequency = self.expected_state_visitation_frequency(user_reward, batch_trajectories) # part 2 of gradient
        #state_visitation_frequency = self.expected_state_visitation_frequency_deterministic(reward_weight)
        
        # normalize expected svf to sum to 1
        norm_expected_svf = state_visitation_frequency / state_visitation_frequency.sum()
        grad = f_delta -feature_map.dot(norm_expected_svf)
        grads.append(np.abs(grad).sum())
        # change reward_weight
        reward_weight += self.learning_rate * grad

        # keep a copy of the old reward
        last_reward = user_reward.copy()

      print("gradient mag: ", np.linalg.norm(grad), " reward diff: ", reward_diff)
      #if i % 10 == 0:
      #  print('finished: {}'.format(i))
      #  if self.verbose:
      #    print("gradient mag: ", np.linalg.norm(grad), " reward diff: ", reward_diff)

    # return the reward and gradients
    user_reward = feature_map.dot(reward_weight)

    return user_reward, grads
