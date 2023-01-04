import argparse
import gym
from agents.dqn import DQN

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='ReinforcementLearning',
        description='Performs reinforcement learning on the Lunar Lander environment'
    )
    parser.add_argument('-a', '--algorithm', default='dqn')
    args = vars(parser.parse_args())
    algorithm = args['algorithm']
    env = gym.make('LunarLander-v2')

    if algorithm == 'dqn':
        print("Running DQN")
        agent = DQN(action_space=env.action_space.n, state_space=env.observation_space.shape[0])
        agent.train(env, 1000)
    elif algorithm == 'ddqn':
        print("Running Double DQN")
        pass
    elif algorithm == 'dqfd':
        print("Running Deep Q-learning from Demonstrations")
        pass