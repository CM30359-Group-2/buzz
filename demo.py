import gym
from keras.models import load_model
from keras.losses import MeanSquaredError
import numpy as np

env = gym.make('LunarLander-v2', render_mode='human')
filename = "model_300.h5"
trained_model = load_model(filename, custom_objects={
    'masked_huber_loss': MeanSquaredError()})

evaluation_max_episodes = 10
evaluation_max_steps = 1000


def get_q_values(model, state):
    input = state[np.newaxis, ...]
    return model.predict(input)[0]


def select_best_action(q_values):
    return np.argmax(q_values)


rewards = []
for episode in range(1, evaluation_max_episodes + 1):
    state = env.reset()

    fraction_finished = 0.0
    state = np.append(state[0], fraction_finished)

    episode_reward = 0

    step = 1

    for step in range(1, evaluation_max_steps + 1):
        env.render()
        q_values = get_q_values(trained_model, state)
        action = select_best_action(q_values)
        new_state, reward, done, info, _ = env.step(action)

        fraction_finished = (step + 1) / evaluation_max_steps
        new_state = np.append(new_state, fraction_finished)

        episode_reward += reward

        if step == evaluation_max_steps:
            done = True

        state = new_state

        if done:
            break

        print(
            f"Episode {episode} finished after {step} steps with reward {episode_reward}")
        rewards.append(episode_reward)

print(f"Average reward: {np.average(rewards)}")
