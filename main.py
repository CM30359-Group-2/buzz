import json
import os
import random
import shutil
import uuid
import gym
import numpy as np
from keras import models

from buffer import ReplayBuffer, Transition
from prioritised_replay_buffer import PrioritisedReplayBuffer
from q_network import QNetwork, masked_huber_loss


env = gym.make('LunarLander-v2')


def calculate_target_values(agent, target_agent, state_transitions: 'list[Transition]', discount_factor):
    new_states = np.array([t.next_state for t in state_transitions])

    q_values = agent.get_multiple_q_values(new_states)
    target_q_values = target_agent.get_multiple_q_values(new_states)

    targets = []
    for index, state_transition in enumerate(state_transitions):
        best_action = select_best_action(q_values[index])
        best_action_q_value = target_q_values[index][best_action]

        if state_transition.done:
            target_value = state_transition.reward
        else:
            target_value = state_transition.reward + discount_factor * best_action_q_value

        target_vector = [0] * 4
        target_vector[state_transition.action] = target_value
        targets.append(target_vector)

    return np.array(targets)


def train_model(model, states, targets):
    model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)


def copy_model(model):
    backup_file = 'backup_'+str(uuid.uuid4())
    model.save(backup_file)
    new_model = models.load_model(backup_file, custom_objects={
                                  'masked_huber_loss': masked_huber_loss(0.0, 1.0)})
    shutil.rmtree(backup_file)
    return new_model


def select_action_epsilon_greedy(q_values, epsilon):
    random_value = random.uniform(0, 1)
    if random_value < epsilon:
        return random.randint(0, len(q_values) - 1)
    else:
        return np.argmax(q_values)


def select_best_action(q_values):
    return np.argmax(q_values)


def dqn(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    replay_buffer = ReplayBuffer(1000, 64, 42)
    agent = QNetwork(env.observation_space.shape[0] + 1, 4)
    target_agent = QNetwork(
        env.observation_space.shape[0] + 1, 4, copy_model(agent.model))
    epsilon = eps_start
    step_count = 0

    for episode in range(n_episodes):
        print(f"Starting episode {episode} with epsilon {epsilon}")

        episode_reward = 0
        state = env.reset()[0]
        fraction_finished = 0.0
        state = np.append(state, fraction_finished)

        for step in range(1, max_t + 1):
            step_count += 1
            q_values = agent.get_q_values(state)
            action = select_action_epsilon_greedy(q_values, epsilon)
            new_state, reward, done, info, _ = env.step(action)

            fraction_finished = (step + 1) / max_t
            new_state = np.append(new_state, fraction_finished)

            episode_reward += reward

            if step == max_t:
                done = True

            state_transition = Transition(
                state, action, reward, new_state, done)
            replay_buffer.add(state_transition)

            state = new_state

            if step_count % 1000 == 0:
                target_agent = QNetwork(
                    env.observation_space.shape[0] + 1, 4, copy_model(agent.model))

            if len(replay_buffer) >= 256 and step_count % 4 == 0:
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
        if episode != 0 and episode % 5 == 0:
            backup_file = f"model_{episode}.h5"
            print(f"Backing up model to {backup_file}")
            agent.model.save(backup_file)

        epsilon *= eps_decay
        epsilon = max(epsilon, eps_end)


def dqfd(n_episodes=2000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, pre_training_updates=40000):
    replay_buffer = PrioritisedReplayBuffer(1000000, 64, 0.6)
    agent = QNetwork(env.observation_space.shape[0], 4)
    target_agent = QNetwork(
        env.observation_space.shape[0], 4, copy_model(agent.model))
    epsilon = eps_start
    beta = 0.4
    step_count = 0

    # Load the expert data
    print("Loading expert data")
    # For all json files in demo directory
    transition_total = 0
    for file in os.listdir("demos"):
        # Load the json file
        with open(os.path.join("demos", file)) as json_file:
            print(f"Loading {file}")
            # For all transitions in the json file
            demo = json.load(json_file)
            for transition in demo["data"]:
                transition_total += 1
                # Create a state transition from the json data
                state_transition = Transition(
                    np.array(transition["obs_t"]), transition["action"], transition["rew"], np.array(transition["obs_tp1"]), transition["terminated"])
                # Add the transition to the replay buffer
                replay_buffer.add(state_transition)

    print(f"Loaded {transition_total} transitions")

    # Pre-train the model
    print("Pre-training model")
    for step in range(pre_training_updates):
        batch = replay_buffer.sample(beta)
        targets = calculate_target_values(
            agent, target_agent, batch, 0.99)
        states = np.array(
            [state_transition.state for state_transition in batch])
        train_model(agent.model, states, targets)

        if step % 1000 == 0:
            # Update the target network
            print(f"Updating target network after {step} steps")
            target_agent = QNetwork(
                env.observation_space.shape[0] + 1, 4, copy_model(agent.model))

    print("Finished pre-training")

    for episode in range(n_episodes):
        print(f"Starting episode {episode} with epsilon {epsilon}")

        episode_reward = 0
        state = env.reset()[0]
        fraction_finished = 0.0
        state = np.append(state, fraction_finished)

        for step in range(1, max_t + 1):
            step_count += 1
            q_values = agent.get_q_values(state)
            action = select_action_epsilon_greedy(q_values, epsilon)
            new_state, reward, done, info, _ = env.step(action)

            fraction_finished = (step + 1) / max_t
            new_state = np.append(new_state, fraction_finished)

            episode_reward += reward

            if step == max_t:
                done = True

            state_transition = Transition(
                state, action, reward, new_state, done)
            replay_buffer.add(state_transition)

            state = new_state

            if step_count % 1000 == 0:
                target_agent = QNetwork(
                    env.observation_space.shape[0] + 1, 4, copy_model(agent.model))

            if len(replay_buffer) >= 256 and step_count % 4 == 0:
                batch = replay_buffer.sample(beta)
                targets = calculate_target_values(
                    agent, target_agent, batch, 0.99)
                states = np.array(
                    [state_transition.state for state_transition in batch])
                train_model(agent.model, states, targets)

            if done:
                break

        print(
            f"Episode {episode} finished after {step} steps with reward {episode_reward}")
        if episode != 0 and episode % 5 == 0:
            backup_file = f"model_{episode}.h5"
            print(f"Backing up model to {backup_file}")
            agent.model.save(backup_file)

        epsilon *= eps_decay
        epsilon = max(epsilon, eps_end)
        # Linearly increase beta towards 1 at the end of training
        beta = min(1.0, beta + 0.001)


# scores = dqd()
scores = dqfd()
