import numpy as np


def discretize_state(s, ep):
    state = s[0]
    print(ep, state)
    print(type(state))
    discrete_state = (min(2, max(-2, int(state[0] / 0.05))), \
                        min(2, max(-2, int(state[1] / 0.1))), \
                        min(2, max(-2, int(state[2] / 0.1))), \
                        min(2, max(-2, int(state[3] / 0.1))), \
                        min(2, max(-2, int(state[4] / 0.1))), \
                        min(2, max(-2, int(state[5] / 0.1))), \
                        int(state[6]), \
                        int(state[7]))

    return discrete_state

class QLearning:
    def __init__(self, env, alpha = 0.5, gamma = 0.9, epsilon = 1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {}

       

    def train(self):
        for i in range(1000):
            S = discretize_state(self.env.reset(), i)
            #S = self.env.reset()

            while True:
                self.env.render()

                # choose action A from state S using epsilon greedy policy
                A = self.__epsilon_greedy(S)

                # take action A, take immediate reward R and move to state S_
                # S_, reward, terminated, _, _ = self.env.step(A)
                new_S, reward, terminated, _, _ = self.env.step(A)
                S_ = discretize_state(new_S, i)

                # update Q values
                if not terminated:
                    x = self.gamma * self.__greedy(S_)
                    self.Q[(S, A)] += self.alpha * (reward + x - self.Q[(S, A)])
                else:
                    self.Q[(S, A)] += self.alpha * (reward - self.Q[(S, A)])
                    break

                S = S_

    def __greedy(self, state):
        return np.argmax([self.Q[(state, action)] for action in range(4)])

    def __epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(range(self.env.action_space.n))
        else:
            return self.__greedy(state)
