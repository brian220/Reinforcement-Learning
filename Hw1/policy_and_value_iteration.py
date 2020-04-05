# Spring 2020, IOC 5262 Reinforcement Learning
# HW1: Policy Iteration and Value iteration for MDPs
       
import numpy as np
import gym

def get_rewards_and_transitions_from_env(env):
    # Get state and action space sizes
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Intiailize matrices
    R = np.zeros((num_states, num_actions, num_states))
    P = np.zeros((num_states, num_actions, num_states))

    # Get rewards and transition probabilitites for all transitions from an OpenAI gym environment
    for s in range(num_states):
        for a in range(num_actions):
            for transition in env.P[s][a]:
                prob, s_, r, done = transition
                R[s, a, s_] = r
                P[s, a, s_] = prob
    
    return R, P

def value_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """        
        Run value iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for value iteration
            eps: float
                for the termination criterion of value iteration 
        ----------
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize the value function V(s)
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve V(s) using the Bellman optimality operator
            4. Derive the optimal policy using V(s)
        ----------
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_states)])
    print(policy)

    ##### FINISH TODOS HERE #####
    R, P = get_rewards_and_transitions_from_env(env)

    # Initialize k
    k = 0
    # Initialize V(s)
    V = np.zeros(num_states)

    # Iterate and improve V(s)
    while k < max_iterations:
        V_new = np.zeros(num_states)
        delta = 0
        # Update value for each state
        for state in range(len(V)):
            # Check for each actions
            max_value = 0
            for action in range(len(R[state])):
                current_value = 0
                for state_ in range(len(R[state][action])):
                    prob = P[state][action][state_]
                    if prob == 1:
                        # Bellman optimality operator
                        current_value += R[state][action][state_] + gamma * prob * V[state_]
                        if current_value > max_value:
                            max_value = current_value
                        break

            if np.abs(max_value - V[state]) > delta:
                delta = np.abs(max_value - V[state])
            V_new[state] = max_value
        
        if delta < eps:
            break
        V = V_new
        k = k + 1
        print(k, delta)

    # Get the optimal policy
    for state in range(len(V)):
        max_value = 0
        best_action = 0
        for action in range(len(R[state])):
            current_value = 0
            for state_ in range(len(R[state][action])):
                prob = P[state][action][state_]
                if prob == 1:
                    # Bellman optimality operator
                    current_value += (R[state][action][state_] + gamma * prob * V[state_])
                    if current_value > max_value:
                        max_value = current_value
                        best_action = action

        policy[state] = best_action

    #############################
    
    # Return optimal policy    
    return policy

def policy_iteration(env, gamma=0.9, max_iterations=10**6, eps=10**-3):
    """ 
        Run policy iteration (You probably need no more than 30 lines)
        
        Input Arguments
        ----------
            env: 
                the target environment
            gamma: float
                the discount factor for rewards
            max_iterations: int
                maximum number of iterations for the policy evalaution in policy iteration
            eps: float
                for the termination criterion of policy evaluation 
        ----------  
        
        Output
        ----------
            policy: np.array of size (500,)
        ----------
        
        TODOs
        ----------
            1. Initialize with a random policy and initial value function
            2. Get transition probabilities and reward function from the gym env
            3. Iterate and improve the policy
        ----------
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    # Initialize with a random policy
    policy = np.array([env.action_space.sample() for _ in range(num_states)])
    

    ##### FINISH TODOS HERE #####
    R, P = get_rewards_and_transitions_from_env(env)
    # Initialize k
    k = 0
    # Initial value function
    V_PI_K = np.zeros(num_states)

    # Iterate and improve policy
    while k < max_iterations:
        # Policy evaluation
        delta = 0
        V_PI_K_new = np.zeros(num_states)
        for state in range(len(V_PI_K)):
            current_action = policy[state]
            for state_ in range(len(R[state][current_action])):
                prob = P[state][current_action][state_]
                if prob == 1:
                    V_PI_K_new[state] += (R[state][current_action][state_] + gamma * prob * V_PI_K[state_])
                    break

            if np.abs(V_PI_K_new[state] - V_PI_K[state]) > delta:
                delta = np.abs(V_PI_K_new[state] - V_PI_K[state])
            
        if delta < eps:
            break
        V_PI_K = V_PI_K_new

        # Policy Iteration
        policy_stable = True
        new_policy = np.zeros(num_states)

        for state in range(len(V_PI_K)):
            Q_PI_K = np.zeros(num_actions)
            for action in range(len(Q_PI_K)):
                for state_ in range(len(R[state][action])):
                    prob = P[state][action][state_]
                    if prob == 1:
                        Q_PI_K[action] = (R[state][action][state_] + gamma * prob * V_PI_K[state_])
                        break
            
            # Choose the best action
            action = np.argmax(Q_PI_K)
            if action != policy[state]:
                policy_stable = False
            
            policy[state] = action

        if policy_stable:
            break
        k = k + 1
        print(k, policy_stable, delta)

    #############################

    # Return optimal policy
    return policy

def print_policy(policy, mapping=None, shape=(0,)):
    print(np.array([mapping[action] for action in policy]).reshape(shape))


def run_pi_and_vi(env_name):
    """ 
        Enforce policy iteration and value iteration
    """    
    env = gym.make(env_name)
    print('== {} =='.format(env_name))
    print('# of actions:', env.action_space.n)
    print('# of states:', env.observation_space.n)
    print(env.desc)

    vi_policy = value_iteration(env)
    pi_policy = policy_iteration(env)

    return pi_policy, vi_policy
    # return pi_policy
   

if __name__ == '__main__':
    # OpenAI gym environment: Taxi-v2
    pi_policy, vi_policy = run_pi_and_vi('Taxi-v2')
    # pi_policy = run_pi_and_vi('Taxi-v2')

    # For debugging
    action_map = {0: "S", 1: "N", 2: "E", 3: "W", 4: "P", 5: "D"}
    print_policy(pi_policy, action_map, shape=None)
    print_policy(vi_policy, action_map, shape=None)
    
    # Compare the policies obatined via policy iteration and value iteration
    diff = sum([abs(x-y) for x, y in zip(pi_policy.flatten(), vi_policy.flatten())])        
    print('Discrepancy:', diff)
    



