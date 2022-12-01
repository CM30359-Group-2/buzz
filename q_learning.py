import numpy as np
import gym

def clamp_value(value, min_value, max_value, step):
    return min(max_value, max(min_value, int(value) / step))

def discretize_state(s):
    return (clamp_value(s[0], -3, 3, 0.033), # x_position
            clamp_value(s[1], -2, 2, 0.07),  # y_position
            clamp_value(s[2], -2, 2, 0.125), # x_velocity
            clamp_value(s[3], -2, 2, 0.125), # y_velocity
            clamp_value(s[4], -2, 2, 0.125), # angle
            clamp_value(s[5], -2, 2, 0.125), # angular_velocity
            int(state_[6]),                  # leg0_contact
            int(state_[7]))                  # leg1_contact

def init_Q_values():
    Q = {}
    for action in range(0,4):
        for x_position in range(-3,4):
            for y_position in range(-2,3):
                for x_velocity in range(-2,3):
                    for y_velocity in range(-2,3):
                        for angle in range(-2,3):
                            for angular_velocity in range(-2,3):
                                for leg0_contact in range(0,2):
                                    for leg1_contact in range(0,2):
                                        state = (x_position, y_position, x_velocity, y_velocity,
                                                angle, angular_velocity, leg0_contact, leg1_contact)
                                        Q[(state, action)] = 0

class QLearning:
    def __init__(self, env, alpha = 0.1, gamma = 0.99, epsilon = 1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = init_Q_values()

    def train(self):
        average_return = []
        total = 0.0

        # for 10000 episodes
        for i in range(10000):
            # decay epsilon until it is 0.02 so it searches less each episode and relies
            # more on experience rather than exploration
            self.epsilon = max(self.epsilon * 0.995, 0.02)

            # not sure if this line is important
            # self.env = gym.make("LunarLander-v2", continuous=False)

            S = discretize_state(self.env.reset()[0])

            # for 1000 steps in the episode
            for _ in range(1000):
                self.env.render()

                # choose action A from state S using epsilon greedy policy
                A = self.__epsilon_greedy(S)

                # take the action in the environment and observe the new state
                new_S, reward, terminated, _, _ = self.env.step(A)
                S_ = discretize_state(new_S)

                total += reward

                # update Q values
                if not terminated:
                    self.Q[(S, A)] += self.alpha * (reward + self.gamma * self.__epsilon_greedy(S_) - self.Q[(S, A)])
                else:
                    self.Q[(S, A)] += self.alpha * (reward - self.Q[(S, A)])
                    break

                S = S_

            # accumulate average score over 100 episodes
            if i % 100 == 0 and i > 0:
                average_return.append(total / 100)
                total = 0.0

        return average_return

    def __greedy(self, state):
        return max([self.Q[(state, action)] for action in range(4)])

    def __epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(range(4))
        else:
            return np.argmax([self.Q[(state, action)] for action in range(4)])
