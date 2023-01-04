import os
import random
import uuid

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

    def __init__(self, action_space, state_space):
        Agent.__init__(self, action_space, ReplayBuffer(self.memory_size, self.batch_size, self.seed))
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

        target_q_values = rewards + self.gamma * np.amax(self.target.predict_on_batch(next_states),axis=1) * (1 - dones)
        q_values = self.policy.predict_on_batch(states)
        indices = np.array([i for i in range(self.batch_size)])
        q_values[[indices], [actions]] = target_q_values

        self.policy.fit(states, q_values, epochs=1, verbose=0)
        

def dqn(env, n_episodes=2000, max_t=3000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    script_dir = os.path.dirname(__file__)
    replay_buffer = ReplayBuffer(1000000, 64, 42)
    agent = QNetwork(env.observation_space.shape[0], 4)
    target_agent = QNetwork(
        env.observation_space.shape[0], 4, copy_model(agent.model))
    epsilon = eps_start
    step_count = 0

    for episode in range(n_episodes):
        print(f"Starting episode {episode} with epsilon {epsilon}")

        episode_reward = 0
        state = env.reset()

        for step in range(1, max_t + 1):
            step_count += 1
            q_values = agent.get_q_values(state)
            action = select_action_epsilon_greedy(q_values, epsilon)
            new_state, reward, done, _ = env.step(action)

            episode_reward += reward

            if step == max_t:
                done = True

            state_transition = Transition(
                state, action, reward, new_state, done)
            replay_buffer.add(state_transition)

            state = new_state

            if step_count % 1000 == 0:
                target_agent = QNetwork(
                    env.observation_space.shape[0], 4, copy_model(agent.model))

            if len(replay_buffer) >= 256:
                batch = replay_buffer.sample()
                targets = calculate_target_values(
                    agent, target_agent, batch, 0.99)
                states = np.array(
                    [state_transition.state for state_transition in batch])
                train_model(agent.model, states, targets)

            if done:
                break

        print(
            f"Episode {episode} finished after {step} steps with reward {episode_reward}")
        if episode != 0 and episode % 50 == 0:
            backup_file = f"dqn{episode}.h5"
            print(f"Backing up model to {backup_file}")
            agent.model.save(os.path.join(script_dir, backup_file))

        epsilon *= eps_decay
        epsilon = max(epsilon, eps_end)