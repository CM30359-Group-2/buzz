import os
import random
import uuid
from gym import Env

import numpy as np
from buffer import ReplayBuffer, Transition
from q_network import QNetwork
from agents.agent import Agent
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from utils import calculate_target_values, select_action_epsilon_greedy, copy_model, train_model

class DDQN(Agent):
    batch_size = 64
    memory_size = 1000000
    seed = 0

    def __init__(self, action_space, state_space, checkpoint=False):
        Agent.__init__(self, action_space, state_space, ReplayBuffer(self.memory_size, self.batch_size, self.seed))
        self.checkpoint = checkpoint
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_date = 0.001
        self.gamma = 0.99
        self.max_steps = 2000
        self.policy = self.build_model()
        self.target = self.build_model()

        self.update_target()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_space, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target(self):
        return self.target.set_weights(self.policy.get_weights())

    def act(self, state):
        # Epsilon greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        else:
            action_values = self.policy.predict(state)
            return np.argmax(action_values[0])

    def replay(self):
        """
        Replay the memory and train the policy network

        :return: None
        """
        if len(self.memory) < self.batch_size:
            return

        mini_batch = self.memory.sample()
        states = np.array([transition.state for transition in mini_batch])
        next_states = np.array([transition.next_state for transition in mini_batch])
        rewards = np.array([transition.reward for transition in mini_batch])
        dones = np.array([transition.done for transition in mini_batch])
        actions = np.array([transition.action for transition in mini_batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        # Calculate the target q-values
        target_q_values = rewards + self.gamma * np.amax(self.target.predict_on_batch(next_states), axis=1) * (1 - dones)
        q_values = self.policy.predict_on_batch(states)
        q_values[range(self.batch_size), actions] = target_q_values

        # Train the policy network to better approximate the target q-values
        self.policy.fit(states, q_values, epochs=1, verbose=0)
        

    def train(self, env: Env, episodes=1000):
        env.seed(self.seed)
        np.random.seed(self.seed)
        rewards = []

        for episode in range(episodes):
            print(f"Starting episode {episode} with epsilon {self.epsilon}")

            episode_reward = 0
            state = env.reset()
            state = np.reshape(state, [1, 8])

            for step in range(1, self.max_steps + 1):
                action = self.act(state)
                new_state, reward, done, _ = env.step(action)
                new_state = np.reshape(new_state, (1, 8))

                episode_reward += reward

                state_transition = Transition(
                    state, action, reward, new_state, done)
                self.remember(state_transition)

                state = new_state
                self.replay()

                if done:
                    break

            self.update_target()
            rewards.append(episode_reward)
            print(f"{episode}/{episodes}: {step} steps with reward {episode_reward}")

            running_mean = np.mean(rewards[-100:])
            if running_mean > 200:
                print(f"Solved after {episode} episodes with reward {running_mean}")
                break
            
            print(f"Average over last 100 episodes: {running_mean}")
            if episode != 0 and episode % 50 == 0 and self.checkpoint:
                self.save_model(episode)         

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def save_model(self, episode: int):
        script_dir = os.path.dirname(__file__)
        backup_file = f"ddqn_{episode}.h5"
        self.policy.save(os.path.join(script_dir, backup_file))