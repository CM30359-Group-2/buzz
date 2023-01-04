import json
import os
import random
from gym import Env

import numpy as np
from agents.agent import Agent
from buffer import ReplayBuffer
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from agents.transition import Transition

class DQFD(Agent):
    batch_size = 64
    memory_size = 1000000
    seed = 42

    def __init__(self, action_space, state_space, pre_training_epochs=40000, checkpoint=False):
        Agent.__init__(self, action_space, state_space, ReplayBuffer(self.memory_size, self.batch_size, self.seed))
        self.pre_training_epochs = pre_training_epochs
        self.checkpoint = checkpoint
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
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def act(self, state):
        # Epsilon greedy
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        else:
            action_values = self.model.predict(state)
            return np.argmax(action_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        mini_batch = self.memory.sample()
        states = np.array([transition.state for transition in mini_batch])
        next_states = np.array([transition.next_state for transition in mini_batch])
        rewards = np.array([transition.reward for transition in mini_batch])
        dones = np.array([transition.done for transition in mini_batch])
        actions = np.array([transition.action for transition in mini_batch])

        # print(states)
        states = np.squeeze(states)
        next_states = np.squeeze(next_states)

        target_q_values = rewards + self.gamma * np.amax(self.model.predict_on_batch(next_states), axis=1) * (1 - dones)
        q_values = self.model.predict_on_batch(states)
        indices = np.array([i for i in range(self.batch_size)])
        q_values[[indices], [actions]] = target_q_values

        self.model.fit(states, q_values, epochs=1, verbose=0)

    def pre_train(self):
        script_dir = os.path.dirname(__file__)
        print("Loading expert data")
        total_transitions = 0
        
        demos_path = os.path.join(script_dir, '../demos')
        for file in os.listdir(demos_path):
            with open(os.path.join(demos_path, file)) as demo_file:
                print(file)

                demo = json.load(demo_file)
                for transition in demo["data"]:
                    total_transitions += 1
                    states = np.reshape(np.array(transition["obs_t"], dtype=np.float32), (1,8))
                    new_states = np.reshape(np.array(transition["obs_tp1"], dtype=np.float32), (1,8))

                    parsed_transition = Transition(
                        states, transition["action"], transition["rew"], new_states, transition["terminated"])
                    if parsed_transition.state.shape != (1,8):
                        print(parsed_transition.state.shape)
                    self.remember(parsed_transition)
            break
        
        print(f"Loaded {total_transitions} transitions")

        for _ in range(self.pre_training_epochs):
            self.replay()

        print("Finished pre-training")


    def train(self, env: Env, episodes=1000):
        env.reset(seed=self.seed)
        np.random.seed(self.seed)
        rewards = []

        if self.pre_training_epochs > 0:
            print("Pre-training")
            self.pre_train()
        else:
            print("Skipping pre-training")

        for episode in range(episodes):
            print(f"Starting episode {episode} with epsilon {self.epsilon}")

            episode_reward = 0
            state = env.reset()
            state = np.reshape(state, (1,8))

            for step in range(1, self.max_steps + 1):
                action = self.act(state)
                new_state, reward, done, _ = env.step(action)
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
            if episode != 0 and episode % 50 == 0 and self.checkpoint:
                self.save_model(episode)         

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return rewards
               

    def save_model(self, episode: int):
        script_dir = os.path.dirname(__file__)
        backup_file = f"dqn{episode}.h5"
        print(f"Backing up model to {backup_file}")
        self.model.save(os.path.join(script_dir, backup_file))


    def load(self):
        pass