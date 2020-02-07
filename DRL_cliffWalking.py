import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

# world height
WORLD_HEIGHT = 4

# world width
WORLD_WIDTH = 12

# probability for exploration
EPSILON = 0.1

# step size
ALPHA = 0.5

# gamma for Q-Learning and Expected Sarsa
GAMMA = 1

# all possible actions
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTIONS = [ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT]
action_count = np.zeros(4)

# initial state action pair values
START = [3, 0]
GOAL = [3, 11]

# time variable
t = 0

# UCB variable
UCB_param = 5
UCB = False

def step(state, action):
    action_count[action] += 1
    i, j = state
    if action == ACTION_UP:
        next_state = [max(i - 1, 0), j]
    elif action == ACTION_LEFT:
        next_state = [i, max(j - 1, 0)]
    elif action == ACTION_RIGHT:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == ACTION_DOWN:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    else:
        assert False

    reward = -1
    if (action == ACTION_DOWN and i == 2 and 1 <= j <= 10) or (
        action == ACTION_RIGHT and state == START):
        reward = -100
        next_state = START

    return next_state, reward

# choose an action based on epsilon greedy algorithm
def choose_action(state, q_value):
    if UCB:
        UCB_estimation = q_value[state[0], state[1], :] + \
                UCB_param * np.sqrt(np.log(t + 1) / (action_count + 1e-5))
        q_best = np.max(UCB_estimation)
        return np.random.choice(np.where(UCB_estimation == q_best)[0])

    if np.random.binomial(1, EPSILON) == 1:
        return np.random.choice(ACTIONS)
    else:
        values_ = q_value[state[0], state[1], :]
        return np.random.choice([action_ for action_, value_ in enumerate(values_) if value_ == np.max(values_)])

# an episode with Sarsa
# @q_value: values for state action pair, will be updated
# @expected: if True, will use expected Sarsa algorithm
# @step_size: step size for updating
# @return: total rewards within this episode
def sarsa(q_value, expected=False, step_size=ALPHA):
    state = START
    action = choose_action(state, q_value)
    rewards = 0.0
    t = 0
    action_count = np.zeros(4)
    while state != GOAL:
        next_state, reward = step(state, action)
        next_action = choose_action(next_state, q_value)
        rewards += reward
        if not expected:
            target = q_value[next_state[0], next_state[1], next_action]
        else:
            # calculate the expected value of new state
            target = 0.0
            q_next = q_value[next_state[0], next_state[1], :]
            best_actions = np.argwhere(q_next == np.max(q_next))
            for action_ in ACTIONS:
                if action_ in best_actions:
                    target += ((1.0 - EPSILON) / len(best_actions) + EPSILON / len(ACTIONS)) * q_value[next_state[0], next_state[1], action_]
                else:
                    target += EPSILON / len(ACTIONS) * q_value[next_state[0], next_state[1], action_]
        target *= GAMMA
        q_value[state[0], state[1], action] += step_size * (
                reward + target - q_value[state[0], state[1], action])
        state = next_state
        action = next_action
        t += 1
    return rewards

# an episode with Q-Learning
# @q_value: values for state action pair, will be updated
# @step_size: step size for updating
# @return: total rewards within this episode
def q_learning(q_value, step_size=ALPHA):
    state = START
    rewards = 0.0
    t = 0
    action_count = np.zeros(4)
    while state != GOAL:
        action = choose_action(state, q_value)
        next_state, reward = step(state, action)
        rewards += reward
        # Q-Learning update
        q_value[state[0], state[1], action] += step_size * (
                reward + GAMMA * np.max(q_value[next_state[0], next_state[1], :]) -
                q_value[state[0], state[1], action])
        state = next_state
        t += 1
    return rewards

# print optimal policy
def print_optimal_policy(q_value):
    optimal_policy = []
    for i in range(0, WORLD_HEIGHT):
        optimal_policy.append([])
        for j in range(0, WORLD_WIDTH):
            if [i, j] == GOAL:
                optimal_policy[-1].append('G')
                continue
            bestAction = np.argmax(q_value[i, j, :])
            if bestAction == ACTION_UP:
                optimal_policy[-1].append('U')
            elif bestAction == ACTION_DOWN:
                optimal_policy[-1].append('D')
            elif bestAction == ACTION_LEFT:
                optimal_policy[-1].append('L')
            elif bestAction == ACTION_RIGHT:
                optimal_policy[-1].append('R')
    for row in optimal_policy:
        print(row)

# Use multiple runs instead of a single run and a sliding window
# With a single run I failed to present a smooth curve
# However the optimal policy converges well with a single run
# Sarsa converges to the safe path, while Q-Learning converges to the optimal path
def runExperiment():
    # episodes of each run
    episodes = 500

    # perform 40 independent runs
    runs = 50

    # epsilon values
    epsilons = [0.5, 0.2, 0.1, 0.05]

    # ucb c values
    ucb_c = [0.1, 0.5, 1, 5, 10]

    UCB = True

    for num, epsilon in enumerate(ucb_c):
        EPSILON = epsilon
        sarsa_str = "rewards_sarsa_e%d = np.zeros(episodes)" % (num)
        exec(sarsa_str)
        q_learning_str = "rewards_q_learning_e%d = np.zeros(episodes)" % (num)
        exec(q_learning_str)
        for r in tqdm(range(runs)):
            sarsa_str = "q_sarsa_e%d = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))" % (num)
            exec(sarsa_str)
            q_learning_str = "q_q_learning_e%d = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))" % (num)
            exec(q_learning_str)
            for i in range(0, episodes):
                # cut off the value by -100 to draw the figure more elegantly
                sarsa_str = "rewards_sarsa_e%d[i] += max(sarsa(q_sarsa_e%d), -100)" % (num, num)
                exec(sarsa_str)
                q_learning_str = "rewards_q_learning_e%d[i] += max(q_learning(q_q_learning_e%d), -100)" % (num, num)
                exec(q_learning_str)
        # averaging over independt runs
        sarsa_str = "rewards_sarsa_e%d /= runs" % (num)
        exec(sarsa_str)
        q_learning_str = "rewards_q_learning_e%d /= runs" % (num)
        exec(q_learning_str)

        # draw reward curves
        sarsa_str = "plt.plot(rewards_sarsa_e%d, label='Sarsa_e%d')" % (num, num)
        exec(sarsa_str)
        q_learning_str = "plt.plot(rewards_q_learning_e%d, label='Q-Learning_e%d')" % (num, num)
        exec(q_learning_str)
    
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.ylim([-100, 0])
    plt.legend()

    plt.savefig('DRL_cliffWalkingucb.png')
    plt.close()

    # display optimal policy
    for num, epsilon in enumerate(ucb_c):
        print('Sarsa(e' + str(num) + ' Optimal Policy:')
        sarsa_str = "print_optimal_policy(q_sarsa_e%d)" % (num)
        exec(sarsa_str)
        print('Q-Learning(e' + str(num) + ' Optimal Policy:')
        q_learning_str = "print_optimal_policy(q_q_learning_e%d)" % (num)
        exec(q_learning_str)

if __name__ == '__main__':
    runExperiment()