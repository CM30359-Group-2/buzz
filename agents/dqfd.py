import os

import numpy as np
from utils import calculate_target_values, select_action_epsilon_greedy, copy_model, train_model
import json
from buffer import Transition
from prioritised_replay_buffer import PrioritisedReplayBuffer
from q_network import QNetwork

def dqfd(env, n_episodes=2000, max_t=3000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, pre_training_updates=40000):
    script_dir = os.path.dirname(__file__)
    replay_buffer = PrioritisedReplayBuffer(500000, 64, 0.6)
    agent = QNetwork(env.observation_space.shape[0], 4)
    target_agent = QNetwork(
        env.observation_space.shape[0], 4, copy_model(agent.model))
    epsilon = eps_start
    beta = 0.4
    step_count = 0

    if pre_training_updates > 0:
        # Load the expert data
        print("Loading expert data")
        # For all json files in demo directory
        transition_total = 0

        rel_path = "../demos"
        abs_file_path = os.path.join(script_dir, rel_path)
        print(abs_file_path)

        for file in os.listdir(abs_file_path):
            # Load the json file
            with open(os.path.join(abs_file_path, file)) as json_file:
                print(f"Loading {file}")
                # For all transitions in the json file
                demo = json.load(json_file)
                for transition in demo["data"]:
                    transition_total += 1

                    # Create a state transition from the json data
                    state_transition = Transition(
                        transition["obs_t"], transition["action"], transition["rew"], transition["obs_tp1"], transition["terminated"])
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
                    env.observation_space.shape[0], 4, copy_model(agent.model))

        print("Finished pre-training")
    else:
        print("Skipping pre-training")

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
        if episode != 0 and episode % 50 == 0:
            backup_file = f"dqfd_{episode}.h5"
            print(f"Backing up model to {backup_file}")
            agent.model.save(os.path.join(script_dir, backup_file))

        # Linear annealing
        epsilon = max(epsilon * eps_decay, eps_end)