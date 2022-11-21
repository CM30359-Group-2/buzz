import numpy as np


def discretize_state(s):
    return (min(1.5, max(-1.5, int(s[0] / 0.1))),   # x_position
            min(1.5, max(-1.5, int(s[1] / 0.1))),   # y_position
            min(5.0, max(-5.0, int(s[2] / 0.1))),   # x_velocity
            min(5.0, max(-5.0, int(s[3] / 0.1))),   # y_velocity
            min(3.14, max(-3.14, int(s[4] / 0.1))), # angle
            min(5.0, max(-5.0, int(s[5] / 0.1))),   # angular_velocity
            int(s[6]),                              # leg0_contact
            int(s[7]))                              # leg1_contact

def init_Q_values():
    Q = {}
    for action in range(4):
        for x_position in range(-1.5, 1.5, 0.1):
            for y_position in range(-1.5, 1.5, 0.1):
                for x_velocity in range(-1.5, 1.5, 0.1):
                    for y_velocity in range(-1.5, 1.5, 0.1):
                        for angle in range(-1.5, 1.5, 0.1):
                            for angular_velocity in range(-1.5, 1.5, 0.1):
                                for leg0_contact in range(0, 2):
                                    for leg1_contact in range(0, 2):
                                        state = (x_position, y_position, x_velocity, y_velocity, angle, angular_velocity, leg0_contact, leg1_contact)
                                        Q[(state, action)] = 0

    return Q


class QLearning:
    def __init__(self, env, alpha = 0.5, gamma = 0.9, epsilon = 1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = range(4)
        self.Q = init_Q_values()

    def train(self):
        for i in range(100000):

            S = discretize_state(self.env.reset()[0])

            while True:
                self.env.render()

                # choose action A from state S using epsilon greedy policy
                A = self.__epsilon_greedy(S)

                # take action A, take immediate reward R and move to state S_
                # S_, reward, terminated, _, _ = self.env.step(A)
                new_S, reward, terminated, _, _ = self.env.step(A)

                S_ = discretize_state(new_S)

                # update Q values
                if not terminated:
                    x = self.gamma * self.__greedy(S_)
                    self.Q[(S, A)] += self.alpha * (reward + x - self.Q[(S, A)])
                else:
                    self.Q[(S, A)] += self.alpha * (reward - self.Q[(S, A)])
                    break

                S = S_

    def __greedy(self, state):
        return np.argmax([self.Q[(state, action)] for action in self.actions])

    def __epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(range(4))
        else:
            return self.__greedy(state)
