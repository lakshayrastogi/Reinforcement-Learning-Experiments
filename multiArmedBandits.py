import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange

matplotlib.use('Agg')


class Bandit:
    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations
    # @sample_averages: if True, use sample averages to update estimations instead of constant step size
    # @UCB_param: if not None, use UCB algorithm to select action
    # @gradient: if True, use gradient based bandit algorithm
    # @gradient_baseline: if True, use average reward as baseline for gradient based bandit algorithm
    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1, sample_averages=False, UCB_param=None,
                 gradient=False, gradient_baseline=False, true_reward=0., stdDev=1):
        self.k = k_arm
        self.step_size = step_size
        self.sample_averages = sample_averages
        self.indices = np.arange(self.k)
        self.time = 0
        self.UCB_param = UCB_param
        self.gradient = gradient
        self.gradient_baseline = gradient_baseline
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.stdDev = stdDev

    def reset(self):
        # real reward for each action
        #self.q_true = np.random.randn(self.k) + self.true_reward
        self.q_true = np.random.normal(scale=self.stdDev, size=self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        # # of chosen times for each action
        self.action_count = np.zeros(self.k)

        self.best_action = np.argmax(self.q_true)

        self.time = 0

    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        if self.UCB_param is not None:
            UCB_estimation = self.q_estimation + \
                self.UCB_param * np.sqrt(np.log(self.time + 1) / (self.action_count + 1e-5))
            q_best = np.max(UCB_estimation)
            return np.random.choice(np.where(UCB_estimation == q_best)[0])

        if self.gradient:
            exp_est = np.exp(self.q_estimation)
            self.action_prob = exp_est / np.sum(exp_est)
            return np.random.choice(self.indices, p=self.action_prob)

        q_best = np.max(self.q_estimation)
        return np.random.choice(np.where(self.q_estimation == q_best)[0])

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        #reward = np.random.randn() + self.q_true[action]
        reward = np.random.normal(scale=self.stdDev) + self.q_true[action]
        self.time += 1
        self.action_count[action] += 1
        self.average_reward += (reward - self.average_reward) / self.time

        if self.sample_averages:
            # update estimation using sample averages
            self.q_estimation[action] += (reward - self.q_estimation[action]) / self.action_count[action]
        elif self.gradient:
            one_hot = np.zeros(self.k)
            one_hot[action] = 1
            if self.gradient_baseline:
                baseline = self.average_reward
            else:
                baseline = 0
            self.q_estimation += self.step_size * (reward - baseline) * (one_hot - self.action_prob)
        else:
            # update estimation with constant step size
            self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward


def simulate(runs, time, bandits):
    rewards = np.zeros((len(bandits), runs, time))
    best_action_counts = np.zeros(rewards.shape)
    for i, bandit in enumerate(bandits):
        for r in trange(runs):
            bandit.reset()
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1
    mean_best_action_counts = best_action_counts.mean(axis=1)
    mean_rewards = rewards.mean(axis=1)
    return mean_best_action_counts, mean_rewards

def runExperiment(runs=2000, time=1000, stdDev=1):
    epsilons = [0, 0.1, 0.01, 0.2, 0.5]
    bandits = [Bandit(epsilon=eps, sample_averages=True, stdDev=stdDev) for eps in epsilons]
    ucb_params = [5, 2, 1, 0.5, 0.1]
    bandits_ucb = [Bandit(epsilon=0, UCB_param=ucb_param, sample_averages=True, stdDev=stdDev) for ucb_param in ucb_params]

    best_action_counts1, rewards1 = simulate(runs, time, bandits)
    best_action_counts2, rewards2 = simulate(runs, time, bandits_ucb)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    for eps, reward in zip(epsilons, rewards1):
        plt.plot(reward, label='epsilon = %.02f' % (eps))
    for ucb_param, reward in zip(ucb_params, rewards2):
        plt.plot(reward, label='UCB c = %.02f' % (ucb_param))
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps, counts in zip(epsilons, best_action_counts1):
        plt.plot(counts, label='epsilon = %.02f' % (eps))
    for ucb_param, counts in zip(ucb_params, best_action_counts2):
        plt.plot(counts, label='UCB c = %.02f' % (ucb_param))
    plt.xlabel('Steps')
    plt.ylabel('Percentage Optimal Action')
    plt.legend()

    fileName = 'RL_Assignment1_stdDev_' + str(stdDev) + '.png'
    plt.savefig(fileName)
    plt.close()

if __name__ == '__main__':
	runExperiment()
	runExperiment(stdDev=10)