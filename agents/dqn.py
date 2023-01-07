import os
import random
from gym import Env

import numpy as np
from agents.agent import Agent
from memory.buffer import ReplayBuffer
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Nadam
from keras.models import load_model

from agents.transition import Transition

class DQN(Agent):
    batch_size = 64
    memory_size = 1000000
    seed = 42

    def __init__(self, action_space, state_space):
        Agent.__init__(self, action_space, state_space, ReplayBuffer(self.memory_size, self.batch_size, self.seed))
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.gamma = 0.99
        self.max_steps = 2000
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation='relu'))
        model.add(Dense(120, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Nadam(learning_rate=self.learning_rate))
        return model

    def choose_action(self, state):
        # Epsilon greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        mini_batch = self.memory.sample_by_idxs()
        states = np.array([transition.state for transition in mini_batch])
        next_states = np.array([transition.next_state for transition in mini_batch])
        rewards = np.array([transition.reward for transition in mini_batch])
        dones = np.array([transition.done for transition in mini_batch])
        actions = np.array([transition.action for transition in mini_batch])

        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        target_q_values = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - dones)
        q_values = self.model.predict_on_batch(states)
        indices = np.array([i for i in range(self.batch_size)])
        q_values[[indices], [actions]] = target_q_values

        self.model.fit(states, q_values, epochs=1, verbose=0)

    def train(self, env: Env, episodes: int, checkpoint: bool, render: bool) -> "list[float]":
        env.reset(seed=self.seed)
        np.random.seed(self.seed)
        rewards = []

        for episode in range(episodes):
            print(f"Starting episode {episode} with epsilon {self.epsilon}")

            episode_reward = 0
            state = env.reset()
            state = np.reshape(state, (1,8))

            for step in range(1, self.max_steps + 1):
                action = self.choose_action(state)
                new_state, reward, done, _ = env.step(action)
                if render:
                    env.render('human')

                new_state = np.reshape(new_state, (1,8))

                episode_reward += reward

                state_transition = Transition(
                    state, action, reward, new_state, done)
                self.remember(state_transition)

                state = new_state
                self.replay()

                if done:
                    break
            rewards.append(episode_reward)
            print(f"{episode}/{episodes}: {step} steps with reward {episode_reward}")

            running_mean= np.mean(rewards[-100:])
            if running_mean > 200:
                print(f"Solved after {episode} episodes with reward {running_mean}")
                break
            
            print(f"Average over last 100 episodes: {running_mean}")
            if episode != 0 and episode % 50 == 0 and checkpoint:
                self.save_checkpoint(episode)         

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return rewards

    def play(self, env: Env, episodes: int, checkpoint_path: str, render: bool) -> "list[float]":
        try:
            self.load_checkpoint(checkpoint_path)
        except ImportError:
            print(f"Loading from {checkpoint_path} is not available")
            return
        except IOError:
            print(f"Checkpoint file {checkpoint_path} is invalid")
            return
        
        rewards = []

        for episode in range(1, episodes + 1):
            episode_reward = 0

            state = env.reset()
            state = np.reshape(state, (1, 8))

            for step in range(1, self.max_steps + 1):

                action = np.argmax(self.model.predict(state)[0])
                new_state, reward, done, _ = env.step(action)
                if render:
                    env.render('human')
                
                new_state = np.reshape(new_state, (1,8))

                episode_reward += reward

                state = new_state

                if done:
                    break
        
            rewards.append(episode_reward)
            print(f"{episode}/{episodes}: {step} steps with reward {episode_reward}")
    
        return rewards

    def save_checkpoint(self, episode: int):
        script_dir = os.path.dirname(__file__)
        backup_file = f"dqn_{episode}.h5"
        print(f"Backing up model to {backup_file}")
        self.model.save(os.path.join(script_dir, backup_file))

    def load_checkpoint(self, checkpoint_path: str):
        try:
            self.model = load_model(checkpoint_path)
        except Exception as _:
            raise