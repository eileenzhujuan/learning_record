# This is adapted from https://github.com/ankonzoid/LearningX/blob/master/classical_RL/multiarmed_bandit/multiarmed_bandit.py

# Design and conduct an experiment to demonstrate the
# difficulties that sample-average methods have for nonstationary problems. Use a modified
# version of the 10-armed testbed in which all the q⇤(a) start out equal and then take
# independent random walks (say by adding a normally distributed increment with mean 0
# and standard deviation 0.01 to all the q⇤(a) on each step). Prepare plots like Figure 2.2
# for an action-value method using sample averages, incrementally computed, and another
# action-value method using a constant step-size parameter, alpha = 0.1. Use epsilon = 0.1 and
# longer runs, say of 10,000 steps.

import numpy as np
from numpy import random
from matplotlib import pyplot as plt


class Env:

    def __init__(self, reward_mean):
        self.reward_mean = reward_mean

    def step(self, action):
        return random.randn() + self.reward_mean[action]

class Agent:
    
    def __init__(self, arms, alpha=None, epsilon=0.1):
        self.arms = arms
        self.n = np.zeros(arms)
        self.q = np.zeros(arms)
        self.alpha = alpha
        self.epsilon = epsilon
        self.random_choice = 0
        self.total_n = 0

    def gen_action(self):
        self.total_n += 1
        random_action = random.random() < self.epsilon
        if random_action:
            self.random_choice += 1
            return random.randint(0, self.arms)
        else:
            return random.choice(np.flatnonzero(self.q == self.q.max()))


    def update_q(self, action, reward):
        self.n[action] += 1
        if self.alpha is not None:
            self.q[action] = self.q[action] + self.alpha * (reward - self.q[action])
        else:
            self.q[action] = self.q[action] + 1.0 / self.n[action] * (reward - self.q[action])
        return self.q[action]


def run_reward(arms, true_reward, total_steps, epsilon, alpha=None):
    reward_group = []
    env = Env(true_reward)
    agent = Agent(len(true_reward), alpha, epsilon)
    last_reward = 0
    for step in range(total_steps):
        action = agent.gen_action()
        reward = env.step(action)
        q_star = agent.update_q(action, reward)
        reward_group.append(reward)

    return reward_group


def get_avg_rewards(true_reward, epsilon, alpha, total_steps, total_exps, nonstationary=False):
    arms = len(true_reward)
    avg_rewards = np.zeros(total_steps)
    for i in range(total_exps):
        if nonstationary:
            random_walk = random.normal(0, 0.01, (arms))
            rewards = run_reward(arms, true_reward + random_walk, total_steps, epsilon, alpha)
        else:
            rewards = run_reward(arms, true_reward, total_steps, epsilon, alpha)
        avg_rewards += rewards
    avg_rewards = avg_rewards / float(total_exps)
    return avg_rewards


def draw_figure_2_2(arms):
    true_reward = [0.2, -0.7, 1.54, 0.45, 1.2, -1.6, -0.2, -1.1, 0.75, -0.65]
    total_steps = 1000
    total_exps = 2000
    for epsilon in [0, 0.01, 0.1]:
        avg_rewards = get_avg_rewards(true_reward, epsilon, None, total_steps, total_exps)
        plt.plot(list(range(len(avg_rewards))), avg_rewards, label=f"{epsilon = }")
    plt.legend()
    plt.show()

def exercise_2_5_code(arms):
    true_reward = random.randn(arms)
    true_reward = [-0.97, 3.35, 1.64, 1.04, 0.08, -0.74, 1.55, 0.89, 1.81, -0.18]
    print(f'{true_reward = }')
    total_steps = 10000
    total_exps = 2000
    for alpha in [None, 0.1]:
        avg_rewards = get_avg_rewards(true_reward, 0.1, alpha, total_steps, total_exps, True)
        plt.plot(list(range(len(avg_rewards))), avg_rewards, label=f"{alpha = }")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # draw_figure_2_2(10)
    exercise_2_5_code(10)
