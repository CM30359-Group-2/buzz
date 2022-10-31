from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
env = gym.make('SuperMarioBros-v3', render_mode='human',
               apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
env.reset()
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    done = terminated or truncated
    if done:
        state = env.reset()

env.close()
