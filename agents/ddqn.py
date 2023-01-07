import os
import random
from gym import Env

import numpy as np
from memory.buffer import ReplayBuffer, Transition
from agents.agent import Agent
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import load_model

class DDQN(Agent):
    batch_size = 64
    memory_size = 1000000
    seed = 0

    def __init__(self, action_space, state_space):
        Agent.__init__(self, action_space, state_space, ReplayBuffer(self.memory_size, self.batch_size, self.seed))
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.learning_rate = 0.0001
        self.gamma = 0.99
        self.max_steps = 2000
        self.policy = self.build_model()
        self.target = self.build_model()

        self.update_target()

    def build_model(self):
        model = Sequential()
        model.add(Dense(120, input_dim=self.state_space, activation='relu'))
        model.add(Dense(150, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target(self):
        return self.target.set_weights(self.policy.get_weights())

    def choose_action(self, state):
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

        mini_batch = self.memory.sample_by_idxs()
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
        

    def train(self, env: Env, episodes: int, checkpoint: bool, render: bool) -> "list[float]":
        env.reset(seed=self.seed)
        np.random.seed(self.seed)
        rewards = []

        for episode in range(episodes):
            print(f"Starting episode {episode} with epsilon {self.epsilon}")

            episode_reward = 0
            state = env.reset()
            state = np.reshape(state, (1, 8))

            for step in range(1, self.max_steps + 1):
                action = self.choose_action(state)
                new_state, reward, done, _ = env.step(action)
                if render:
                    env.render('human')

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
                action = np.argmax(self.policy.predict(state)[0])
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
        backup_file = f"ddqn_{episode}.h5"
        self.policy.save(os.path.join(script_dir, backup_file))

    def load_checkpoint(self, checkpoint_path: str):
        try:
            self.policy = load_model(checkpoint_path)
        except Exception as _:
            raise