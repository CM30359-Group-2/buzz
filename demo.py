import os
import gymnasium as gym
from keras.models import load_model
from keras.losses import MeanSquaredError
import numpy as np

script_dir = os.path.dirname(__file__)
env = gym.make('LunarLander-v2', render_mode='human')
filename = os.path.join(script_dir, 'checkpoints/dqn850.h5')
trained_model = load_model(filename, custom_objects={
    'masked_huber_loss': MeanSquaredError()})

evaluation_max_episodes = 10
evaluation_max_steps = 3000


def get_q_values(model, state):
    input = state[np.newaxis, ...]
    return model.predict(input)[0]


def select_best_action(q_values):
    return np.argmax(q_values)


rewards = []
for episode in range(1, evaluation_max_episodes + 1):
    state = env.reset()[0]
    print(state)

    episode_reward = 0

    step = 1

    for step in range(1, evaluation_max_steps + 1):
        env.render()
        q_values = get_q_values(trained_model, state)
        action = select_best_action(q_values)
        new_state, reward, done, info, _ = env.step(action)

        episode_reward += reward

        if step == evaluation_max_steps:
            done = True

        state = new_state

        if done:
            break

        print(
            f"Step {step} finished with reward {episode_reward}")
        rewards.append(episode_reward)

print(f"Average reward: {np.average(rewards)}")
