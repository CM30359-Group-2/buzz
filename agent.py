import random

import numpy as np
from buffer import ReplayBuffer

from q_network import QNetwork


class Agent():
    """An agent interacts with and learns from the environment"""
    GAMMA = 0.99  # discount factor
    BATCH_SIZE = 64  # minibatch size
    UPDATE_EVERY = 4  # how often to update the network
    BUFFER_SIZE = int(1e5)  # replay buffer size

    def __init__(self, state_size, action_size, seed=42, is_eval=False, model_name=""):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size, seed=42)
        self.qnetwork_target = QNetwork(state_size, action_size, seed=42)

        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed=42)

        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, self.GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy."""
        # Evaluate the local network state
        action_values = self.qnetwork_local.model(state, training=False)

        if random.random() > eps:
            return action_values
        else:
            return np.random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target.model(next_states, training=False)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Calculate the expected value from the local model
        Q_expected = self.qnetwork_local.model(states, training=False)

        # Compute loss
        loss = self.qnetwork_local.model.loss(Q_expected, Q_targets)
        # Backpropagation
        self.qnetwork_local.model.optimizer.minimize(loss, self.qnetwork_local.model.trainable_variables)

        # Update target network
        self.soft_update(self.qnetwork_local.model, self.qnetwork_target.model, tau=1e-3)

    def soft_update(self, local_model, target_model, tau):
        # Soft update model parameters.
        # θ_target = τ*θ_local + (1 - τ)*θ_target
        for target_var, local_var in zip(target_model.trainable_variables, local_model.trainable_variables):
            target_var.assign(tau * local_var + (1.0 - tau) * target_var)
        