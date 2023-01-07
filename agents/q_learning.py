import itertools
import os
import random
from typing import Any, Callable, Union
import numpy as np
from agents.agent import Agent
from agents.transition import Transition
import pickle
from memory.q_table import QTable


def clamp_value(value, min_value, max_value, step):
    return min(max_value, max(min_value, int(value / step)))


def discretize_state(s):
    return (clamp_value(s[0], -3, 3, 0.033),  # x_position
            clamp_value(s[1], -2, 2, 0.07),  # y_position
            clamp_value(s[2], -2, 2, 0.125),  # x_velocity
            clamp_value(s[3], -2, 2, 0.125),  # y_velocity
            clamp_value(s[4], -2, 2, 0.125),  # angle
            clamp_value(s[5], -2, 2, 0.125),  # angular_velocity
            int(s[6]),                  # leg0_contact
            int(s[7]))                  # leg1_contact


class QLearning(Agent):
    seed = 42

    def __init__(self, action_space, state_space, checkpoint=False):
        Agent.__init__(self, action_space, state_space, QTable())
        self.alpha = 0.1
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.02
        self.checkpoint = checkpoint
        self.max_steps = 2000

        # Pre-populate the Q-table with all possible states and actions
        self.pre_populate()

    def pre_populate(self):
        # Define the ranges for each part of the state
        x_position = range(-3, 4)
        y_position = range(-2, 3)
        x_velocity = range(-2, 3)
        y_velocity = range(-2, 3)
        angle = range(-2, 3)
        angular_velocity = range(-2, 3)
        leg0_contact = range(0, 2)
        leg1_contact = range(0, 2)

        # Create all combinations of the above
        states = list(itertools.product(
            x_position,
            y_position,
            x_velocity,
            y_velocity,
            angle,
            angular_velocity,
            leg0_contact,
            leg1_contact
            ))
        actions = range(0, 4)
        # Get tuples of all combinations
        keys = list(itertools.product(states, actions))

        # Add all combinations to the Q-table
        for (state, action) in keys:
            # Fill the remainder of the transition with dummy values - we don't need them for Q-learning
            self.memory.add(Transition(state, action, 0, (), False))

    def choose_action(self, state, greedy=False):
        if np.random.random() <= self.epsilon and not greedy:
            # Epsilon greedy policy
            return random.randrange(self.action_space)
        else:
            return np.argmax(self.memory.recall(state))

    def train(self, env, episodes=10000):
        env.reset(seed=self.seed)
        np.random.seed(self.seed)
        rewards = []

        if self.checkpoint:
            script_dir = os.path.dirname(__file__)
            with open(os.path.join(script_dir, "dict"), "rb") as f:
                self.memory = pickle.load(f)

        for episode in range(episodes):
            print(f"Starting episode {episode} with epsilon {self.epsilon}")
            
            episode_reward = 0
            state = env.reset()
            state = discretize_state(state)

            for step in range(self.max_steps):

                #  choose action A from state S using epsilon greedy policy
                action = self.choose_action(state)

                # take the action in the environment and observe the new state
                new_state, reward, done, _ = env.step(action)
                new_state = discretize_state(new_state)

                episode_reward += reward

                current_q_value = self.memory.recall(state, action)
                if not done:
                    future_reward = self.memory.recall(new_state, self.choose_action(new_state, greedy=True))
                    new_q_value = current_q_value +  self.alpha * (reward + self.gamma * future_reward - current_q_value)
                    self.memory.update(Transition(state, action, 0, (), False), new_q_value)
                else:
                    new_q_value = current_q_value + self.alpha * (reward - current_q_value)
                    self.memory.update(Transition(state, action, 0, (), False), new_q_value)
                    break

                state = new_state

            rewards.append(episode_reward)
            print(f"{episode}/{episodes}: {step} steps with reward {episode_reward}")

            running_mean = np.mean(rewards[-100:])
            if running_mean > 200:
                print(f"Solved after {episode} episodes with reward {running_mean}")
                break

            print(f"Average over last 100 episodes: {running_mean}")
            if episode != 0 and episode % 100 == 0 and self.checkpoint:
                self.save_model(episode)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)        
        return rewards

    def save_model(self, episode: int):
        script_dir = os.path.dirname(__file__)
        backup_file = f"q_{episode}.pickle"
        print(f"Backing up Q-table to {backup_file}")
        with open(os.path.join(script_dir, "dict"), "wb") as f:
                    pickle.dump(self.memory, f)

    def play(self, env, checkpoint, episodes=1, render=False, step_callback: Union[Callable[[Any, Any, int, float, bool, Any], None], None]=None):
        # Load the Q-table from the checkpoint
        script_dir = os.path.dirname(__file__)
        with open(os.path.join(script_dir, checkpoint), "rb") as f:
            self.memory = pickle.load(f)

        env.reset(seed=self.seed)
        np.random.seed(self.seed)
        rewards = []

        for episode in range(episodes):
            print(f"Starting episode {episode}")

            episode_reward = 0
            # Need to separate the generation of new state information from the discretisation step
            raw_state = env.reset()
            state = discretize_state(raw_state)

            for step in range(self.max_steps):
                if render:
                    env.render('rgb_array')

                action = self.choose_action(state, greedy=True)
                raw_new_state, reward, done, info = env.step(action)

                episode_reward += reward
                if step_callback != None:
                    step_callback(raw_state, raw_new_state, action, reward, done, info)

                # Separation of the discretisation step
                raw_state = raw_new_state
                state = discretize_state(raw_new_state)
                if done:
                    break
            
            rewards.append(episode_reward)
            print(f"{episode}/{episodes}: {step} steps with reward {episode_reward}")
        
        return rewards
            
