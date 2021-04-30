import numpy as np

from RLEnvironment import UserTaxiEnvironment, TaxiQLearner


# Compares two Q matrices by seeting the number of states which are equal
def compare_Q_matrices(Q1, Q2): 
  # Assumes Q matrix is |states| x |actions| 
  Q1_actions = np.argmax(Q1, axis=1)
  Q2_actions = np.argmax(Q2, axis=1)
  return np.count_nonzero(np.equal(Q1_actions, Q2_actions))


# simulates 100 episodes for the user as an evaluation metric
# returns the percentage of improvement from original return
def simulate_episodes(user_agent, og_return, min_diff, num_episodes=100):
  total_steps, _, _ = user_agent.evaluate(num_episodes)
  improvement_is = np.where(total_steps < og_return - min_diff)
  # percentage of episodes that did better than original return
  # np.where returns a tuple
  return improvement_is[0].shape[0] / num_episodes

def greedy_baseline(min_diff):
  i = 0
  current_abs_diff = sorted_abs_diff[0] # ith smallest abs diff
  # get the [s,{a}] to be changed
  state_changed = abs_diff_dict[current_abs_diff]
  # change Q user [s,a] to optimal Q[s,a]
  current_Q = user_Q.copy()
  current_Q[state_changed, :] = optimal_Q[state_changed, :]

  # evaluate the current Q on the original RL environment
  current_user_agent = TaxiQLearner(rl_env, epsilon = 0.4)
  current_user_agent.Q = current_Q
  current_user_avg_steps = simulate_episodes(current_user_agent, 100)

  # keep track of current user avg_steps
  # list_avg_step_history = [user_avg_steps, current_user_avg_steps]
  # list_Q_diff = [0, current_abs_diff]

  # stop if current return <= user_avg_steps - some constant
  while ((current_user_avg_steps > user_avg_steps - min_diff) and i < len(abs_diff_dict) - 1):
    i += 1
    # keep updating current user Q
    current_abs_diff = sorted_abs_diff[i] # ith smallest abs diff
    # get the [s,a] to be changed
    state_changed = abs_diff_dict[current_abs_diff]
    # change Q user [s,a] to optimal Q[s,a]
    current_Q[state_changed, :] = optimal_Q[state_changed, :]

    # evaluate the current Q on the original RL environment
    current_user_agent = TaxiQLearner(rl_env, epsilon = 0.4)
    current_user_agent.Q = current_Q
    current_user_avg_steps = simulate_episodes(current_user_agent, 100)
    print('iterationL {}, avg steps: {}'.format(i, current_user_avg_steps))
    # list_avg_step_history.append(current_user_avg_steps)
    # list_Q_diff.append(list_Q_diff[-1] + current_abs_diff)
  return current_user_avg_steps, i + 1

# Given a user's suboptimal policy in the form of a Q matrix,
# we search for a more optimal policy also in the form of a Q matrix
# iteration stops when we hit max_iters or if the policy we found 
# improves average return by atleast min_diff across 100 simulations
# Define sparsity as minimal number of optimal action changes from
# user's policy
# percentage_diff is the minimum percentage of evaluation episodes that have shorter length
# this is specific to our RL env where a better agent is one that has short episode length
# penalty is a function that takes in current state and action and returning the output
# of a reward shaping function
def actionable_sparse_Q_learning(rl_env, user_Q, max_iters, min_diff, percentage_diff, penalty, alpha=0.5, gamma=0.5):
  # create an agent for our user
  user_agent_copy = TaxiQLearner(rl_env)
  user_agent_copy.Q = user_Q.copy()
  # average return
  user_agent_copy_return = simulate_episodes_obtain_mean_return(user_agent_copy, num_episodes=500)
  user_agent_copy_win_percent = simulate_episodes(user_agent_copy, user_agent_copy_return, min_diff, num_episodes=500)
  # create an agent for policy improvement
  new_user = TaxiQLearner(rl_env)
  new_user.Q = user_Q.copy()
  new_user.epsilon = 0.2
  # iteration counter
  i = 0
  # percentage of episodes that beat og user's return
  curr_percentage = user_agent_copy_win_percent
  # init state and action
  curr_state = new_user.env.reset()
  while ((curr_percentage < user_agent_copy_win_percent * (1 + percentage_diff))  and i < max_iters):
    ## UPDATE
    # choose action from state using user's policy - E-greedy
    curr_action = new_user.policy(curr_state)
    # take action a and observe r, s'
    next_state, reward, done, info = new_user.env.step(curr_action)
    # modified Q learning
    current_Q = new_user.Q[curr_state][curr_action]
    optimal_action_for_next_state = new_user.policy(next_state)
    user_action = user_agent_copy.policy(curr_state)
    next_state_action_Q_val = new_user.Q[next_state][optimal_action_for_next_state]
    # if action is the same then regular Q update
    if (user_action == curr_action):
      new_user.Q[curr_state][curr_action] = current_Q + alpha * (reward + gamma * next_state_action_Q_val - current_Q)
    # otherwise we penalize the reward because we want to be sparse
    else:
      new_user.Q[curr_state][curr_action] = current_Q + alpha * (reward - penalty(curr_state, curr_action) + gamma * next_state_action_Q_val - current_Q)

    ## EVALUATION
    curr_percentage = simulate_episodes(new_user, user_agent_copy_return, min_diff, num_episodes=500)
    print(curr_percentage)

    curr_state = next_state
    print("Iteration: ", i)
    i += 1
  
  return new_user, curr_percentage, user_agent_copy_win_percent

# simulates 100 episodes for the user as an evaluation metric
# returns the average return for the user's policy
def simulate_episodes_obtain_median_return(user_agent, num_episodes=100):
  total_steps, _, _ = user_agent.evaluate(num_episodes)
  return np.median(total_steps)

# Given a user's suboptimal policy in the form of a Q matrix,
# we search for a more optimal policy also in the form of a Q matrix
# iteration stops when we hit max_iters or if the policy we found 
# improves average return by atleast min_diff across 100 simulations
# Define sparsity as minimal number of optimal action changes from
# user's policy
# goal_reward is the length of episodes we want to achieve
# this is specific to our RL env where a better agent is one that has short episode length
def sparse_Q_learning(rl_env, user_Q, max_iters, goal_reward, alpha=0.5, gamma=0.5, penalty=1, epsilon = 0.2):
  # set random seed for reproducibility
  # create an agent for our user
  user_agent_copy = TaxiQLearner(rl_env)
  user_agent_copy.Q = user_Q.copy()
  print("Evaluating Original User Policy")
  #user_return = simulate_episodes_obtain_median_return(user_agent_copy)
  #print("User Return: {}".format(user_return))
  #current_return = user_return
  current_return = goal_reward + 1
  # create an agent for policy improvement
  new_user = TaxiQLearner(rl_env)
  new_user.Q = user_Q.copy()
  new_user.epsilon = epsilon
  # iteration counter
  i = 0
  # init state and action
  curr_state = new_user.env.reset()
  while ((current_return > goal_reward)  and i < max_iters):
    done = False
    curr_state = new_user.env.reset() # start a new episode    
    j = 0
    while not done: 
        ## UPDATE
        # choose action from state using user's policy - E-greedy
        curr_action = new_user.policy(curr_state)
        # take action a and observe r, s'
        next_state, reward, done, info = new_user.env.step(curr_action)
        # modified Q learning
        current_Q = new_user.Q[curr_state][curr_action]
        optimal_action_for_next_state = new_user.policy(next_state)
        user_action = user_agent_copy.policy(curr_state)
        next_state_action_Q_val = new_user.Q[next_state][optimal_action_for_next_state]
        # if action is the same then regular Q update
        if (user_action == curr_action):
          new_user.Q[curr_state][curr_action] = current_Q + alpha * (reward + gamma * next_state_action_Q_val - current_Q)
        # otherwise we penalize the reward because we want to be sparse
        else:
          new_user.Q[curr_state][curr_action] = current_Q + alpha * (reward - penalty + gamma * next_state_action_Q_val - current_Q)
        j += 1
    print("Last episode length: {}".format(j))
    

    ## EVALUATION
    if i % 10 == 0:
        print("Evaluating Current New Policy")
        current_return = simulate_episodes_obtain_median_return(new_user)
        print("Current Return: {}".format(current_return))

    curr_state = next_state
    print("Iteration: ", i)
    i += 1
  
  return new_user, current_return