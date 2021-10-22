import random
from collections import deque

import gym
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras.optimizer_v2.adam import Adam

env = gym.make('CartPole-v0')
env.reset()

tf.random.set_seed(200)
FILE = "oving8"

class DQN:
    def create_model(self, shape):

        self.model.add(Dense(24, input_dim=shape, activation='relu',
                             kernel_initializer='he_uniform'))
        self.model.add(Dense(env.action_space.n, activation="linear",
                             kernel_initializer='he_uniform'))
        self.model.compile(loss="mse", optimizer=Adam(lr=0.001),
                      metrics=['accuracy'])

    def __init__(self, input_shape):
        self.model = Sequential()
        self.create_model(input_shape)

        self.memory = deque([], maxlen=2000)

        self.train_start = 128
        self.episodes = 1000
        self.shape = input_shape
        self.epsilon = 0.1
        self.min_epsilon = 0.001
        self.epsilon_decay = 0.995
        self.gamma = 0.99

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(self.model.predict(state))

    def decay(self):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.min_epsilon, self.epsilon)

    def prepare_state(self, state):
        return np.reshape(state, [1, self.shape])

    def new_q_val(self, batch, target, next_target):
        for index, (_, action, reward, _, done) in enumerate(batch):
            target[index][action] = self.update_q_func(reward,
                                                       next_target[index], done)

    def update_q_func(self, reward, param, done):
        if done:
            return reward
        else:
            return reward + self.gamma * np.max(param)

    def get_reward(self, done, step, reward):
        if not done or step == env._max_episode_steps - 1:
            return reward
        else:
            return -100

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.train_start:
            return
        minibatch = random.sample(self.memory,
                                  min(len(self.memory), self.shape))
        states = np.zeros((self.shape, self.shape))
        next_states = np.zeros((self.shape, self.shape))
        for index, (state, _, _, next_state, _) in enumerate(minibatch):
            states[index] = state
            next_states[index] = next_state
        target = self.model.predict(states)
        target_next = self.model.predict(next_states)
        self.new_q_val(minibatch, target, target_next)
        self.model.fit(np.array(states), np.array(target),
                       batch_size=self.shape, verbose=0)
        self.decay()

    def train(self):
        scores = []
        for episode in range(self.episodes):
            done = False
            state = self.prepare_state(env.reset())
            step = 0
            while not done:
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = self.prepare_state(next_state)
                reward = self.get_reward(done, step, reward)
                step += 1
                self.remember(state, action, reward, next_state, done)
                state = next_state
            scores.append(step)
            if step == 200:
                print(f"Saving trained model as {FILE}")
                self.model.save(FILE)
            print(
                f"{scores[episode]}  score for ep {episode + 1} epsilon {self.epsilon}")
            self.replay()
        print('Finished training!')

    def test(self):
        self.model = load_model(FILE)
        state = self.prepare_state(env.reset())
        done = False
        score = 0
        while not done:
            env.render()
            action = np.argmax(self.model.predict(state))
            next_state, reward, done, _ = env.step(action)
            state = self.prepare_state(next_state)
            score += 1
        print(f"{score}  score")
        env.close()


if __name__ == '__main__':
    model = DQN(env.observation_space.shape[0])
    #model.train()
    model.test()
