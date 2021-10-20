import math
from typing import Tuple

import gym
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

env = gym.make('CartPole-v0')
env.reset()

# Observation:
# Cart Position | Cart Velocity | Pole Angle | Pole Angular Velocity
# This code uses the pole angle and the pole angular velocity to train


# Left or right
a = env.action_space.n

# Max possible values
high = [env.observation_space.high[2], math.radians(50)]
# Min possible values
low = [env.observation_space.low[2], -math.radians(50)]

# Defining the size of the q table and filling it
size = (6, 12)
q_table = np.zeros(size + (a,))


def get_discrete(state: 'Current state') -> Tuple[int, ...]:
    _, _, angle, angle_velocity = state
    est = KBinsDiscretizer(n_bins=size,
                           encode='ordinal', strategy='uniform')
    est.fit([low, high])
    return tuple(map(int, est.transform([[angle, angle_velocity]])[0]))


def new_q_value(reward: float, new_state: tuple, discount_factor=1) -> float:
    future_optimal_value = np.max(q_table[new_state])
    learned_value = reward + discount_factor * future_optimal_value
    return learned_value


def learning(n: int, min_rate=0.01) -> float:
    return max(min_rate, min(1.0, 1.0 - math.log10((n + 1) / 25)))


def exploration(number: 'Random generated number',
    min_val=0.1) -> float:
    return max(min_val, min(1., 1. - math.log10((number + 1) / 25)))


for j in range(10000):
    if (j % 500 == 0):
        print(j)
    discrete = get_discrete(env.reset())
    done = False
    score = 0
    while not done:
        if np.random.random() < exploration(j):
            # Get random action
            action = env.action_space.sample()
        else:
            # Get action from Q table based on policy
            action = np.argmax(q_table[discrete])

        observation, reward, done, info = env.step(action)
        new_discrete = get_discrete(observation)

        # Update q table
        lr = learning(j)
        current_pos = discrete + (action,)
        current_q = q_table[current_pos]
        new_q = new_q_value(reward, new_discrete)
        q_table[current_pos] = new_q * lr + (1 - lr) * current_q

        discrete = new_discrete
        score += reward
        env.render()

    print(f"Score: {score}")

env.close()
