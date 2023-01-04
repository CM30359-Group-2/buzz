import os

import numpy as np
from agents.agent import Agent
from buffer import ReplayBuffer, Transition
from keras import Sequential
from keras.layers import Dense

class dqn(Agent):
    def __init__(self, action_space, state_space):
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self.build_model()
        self.gamma = 0.99
        self.batch_size = 64
        self.max_steps = 2000

        Agent.__init__(self, action_space, state_space, ReplayBuffer(1000000, self.batch_size, 42))
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation='relu'))
        model.add(Dense(200, activation='relu'))
        model.add(Dense(self.action_space, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        return model

    def act(self, state):
        # Epsilon greedy
        if np.random.rand() <= self.epsilon:
            return self.action_space.sample()
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

        targets = rewards + self.gamma * np.amax(self.model.predict(next_states), axis=1) * (1 - dones)
        targets_full = self.model.predict_on_batch(states)
        indices = np.array([i for i in range(self.batch_size)])
        targets_full[[indices], [actions]] = targets

        self.model.fit(states, targets_full, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(self, env, episodes=1000):
        rewards = []

        for episode in range(episodes):
            print(f"Starting episode {episode} with epsilon {self.epsilon}")

            episode_reward = 0
            state = env.reset()

            for step in range(1, self.max_steps + 1):
                action = self.act(state)
                new_state, reward, done, _ = env.step(action)

                episode_reward += reward

                state_transition = Transition(
                    state, action, reward, new_state, done)
                self.remember(state_transition)

                state = new_state
                self.replay()

                if done:
                    print(f"{episode}/{episodes}: {step} steps with reward {episode_reward}")
                    break
            rewards.append(episode_reward)

            running_mean= np.mean(rewards[-100:])
            if running_mean > 200:
                print(f"Solved after {episode} episodes with reward {running_mean}")
                break
            
            print(f"Average over last 100 episodes: {running_mean}")
            if episode != 0 and episode % 50 == 0:
                self.checkpoint()

    def checkpoint(self, episode: int):
        script_dir = os.path.dirname(__file__)
        backup_file = f"dqn{episode}.h5"
        print(f"Backing up model to {backup_file}")
        self.model.save(os.path.join(script_dir, backup_file))


    def load(self):
        pass