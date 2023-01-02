import os

import numpy as np
from buffer import ReplayBuffer, Transition
from q_network import QNetwork
from utils import calculate_target_values, select_action_epsilon_greedy, copy_model, train_model

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