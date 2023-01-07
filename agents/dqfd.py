import json
import os
import random
from gym import Env

import numpy as np
from agents.agent import Agent
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.models import load_model

from agents.transition import Transition
from memory.pp_replay_buffer import PartitionedPrioritisedReplayBuffer

class DQFD(Agent):
    batch_size = 64
    memory_size = 1000000
    seed = 42

    def __init__(self, action_space, state_space, pre_training_epochs=15000, checkpoint=False):
        Agent.__init__(self, action_space, state_space, PartitionedPrioritisedReplayBuffer(self.memory_size, self.batch_size, self.seed))
        self.pre_training_epochs = pre_training_epochs
        self.checkpoint = checkpoint
        self.epsilon = 0.01
        self.learning_rate = 0.00025
        self.gamma = 0.99
        self.max_steps = 2000
        self.regularisation_factor = 0.0001
        self.model = self.build_model()
        
    def build_model(self):
        model = Sequential()
        model.add(Dense(150, input_dim=self.state_space, activation='relu', kernel_regularizer=l2(self.regularisation_factor)))
        model.add(Dense(120, activation='relu', kernel_regularizer=l2(self.regularisation_factor)))
        model.add(Dense(self.action_space, activation='linear', kernel_regularizer=l2(self.regularisation_factor)))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
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

        mini_batch, weights, indices = self.memory.sample()
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

        td_errors = np.abs(target_q_values - self.model.predict_on_batch(states).take(actions)) + 0.001
        self.memory.update_priorities(indices, td_errors)

        self.model.fit(states, q_values, epochs=1, verbose=1, sample_weight=weights)

    def pre_train(self):
        script_dir = os.path.dirname(__file__)
        print("Loading expert data")
        transitions = list()
        
        demos_path = os.path.join(script_dir, '../demos')
        for file in os.listdir(demos_path):
            with open(os.path.join(demos_path, file)) as demo_file:
                print(file)

                demo = json.load(demo_file)
                for transition in demo["data"]:
                    states = np.reshape(np.array(transition["obs_t"], dtype=np.float32), (1,8))
                    new_states = np.reshape(np.array(transition["obs_tp1"], dtype=np.float32), (1,8))

                    parsed_transition = Transition(
                        states, transition["action"], transition["rew"], new_states, transition["done"])
                    
                    transitions.append(parsed_transition)
                
            
        self.memory.load(transitions)
        print(f"Loaded {len(transitions)} transitions")

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
                action = self.choose_action(state)
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

        return rewards
               

    def save_model(self, episode: int):
        script_dir = os.path.dirname(__file__)
        backup_file = f"dqfd_{episode}.h5"
        print(f"Backing up model to {backup_file}")
        self.model.save(os.path.join(script_dir, backup_file))


    def play(self, env: Env, checkpoint, episodes=1, render=False):
        script_dir = os.path.dirname(__file__)
        try:
            checkpoint_path = os.path.join(script_dir, checkpoint)
            self.policy = load_model(checkpoint_path)
        except ImportError:
            print(f"Loading from {checkpoint} is not available")
            return
        except IOError:
            print(f"Checkpoint file {checkpoint} is invalid")
            return
        
        rewards = []

        for episode in range(1, episodes + 1):
            episode_reward = 0

            state = env.reset()
            state = np.reshape(state, (1, 8))

            for step in range(1, self.max_steps + 1):
                if render:
                    env.render('rgb_array')

                action = np.argmax(self.policy.predict(state)[0])
                new_state, reward, done, _ = env.step(action)
                new_state = np.reshape(new_state, (1,8))

                episode_reward += reward

                state = new_state

                if done:
                    break
        
            rewards.append(episode_reward)
            print(f"{episode}/{episodes}: {step} steps with reward {episode_reward}")
    
        return rewards

