import numpy as np
import gym
import pickle
import matplotlib.pyplot as plt

from demo import game_callback


def clamp_value(value, min_value, max_value, step):
    return min(max_value, max(min_value, int(value / step)))


def discretize_state(s):
    return (clamp_value(s[0], -3, 3, 0.033),  # x_position
            clamp_value(s[1], -2, 2, 0.07),  # y_position
            clamp_value(s[2], -2, 2, 0.125),  # x_velocity
            clamp_value(s[3], -2, 2, 0.125),  # y_velocity
            clamp_value(s[4], -2, 2, 0.125),  # angle
            clamp_value(s[5], -2, 2, 0.125),  # angular_velocity
            int(s[6]),                  # leg0_contact
            int(s[7]))                  # leg1_contact


class QLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

        for action in range(0, 4):
            for x_position in range(-3, 4):
                for y_position in range(-2, 3):
                    for x_velocity in range(-2, 3):
                        for y_velocity in range(-2, 3):
                            for angle in range(-2, 3):
                                for angular_velocity in range(-2, 3):
                                    for leg0_contact in range(0, 2):
                                        for leg1_contact in range(0, 2):
                                            state = (x_position, y_position, x_velocity, y_velocity,
                                                     angle, angular_velocity, leg0_contact, leg1_contact)
                                            self.Q.update(
                                                {(state, action): 0.0})

    def train(self):

        average_return = []
        total_hundred = 0.0
        y = []
        while True:
            which_dict = input("Use saved Dict(S), or make a new Dict(N)")
            if which_dict.lower() == "s":
                self.Q = pickle.load(open("dict", "rb"))
                print(self.Q)
                break
            elif which_dict.lower() == "n":
                break

        rendering = int(input("How often do you want to render?"))
        train_ep = int(input("How many episodes do you want to train for?"))

        rolling = np.zeros(100)
        # for 10000 episodes
        for i in range(train_ep):
            total_ep = 0.0
            if i % rendering == 0:
                self.env = gym.make(
                    "LunarLander-v2", render_mode="human", continuous=False)
            else:
                self.env = env = gym.make("LunarLander-v2", continuous=False)
            # decay epsilon until it is 0.02 so it searches less each episode and relies
            # more on experience rather than exploration
            self.epsilon = max(self.epsilon * 0.995, 0.02)

            S = discretize_state(self.env.reset()[0])

            # for 1000 steps in the episode
            for _ in range(1000):
                self.env.render()

                #  choose action A from state S using epsilon greedy policy
                A = self.__epsilon_greedy(S)

                # take the action in the environment and observe the new state
                new_S, reward, terminated, _, _ = self.env.step(A)
                S_ = discretize_state(new_S)

                total_hundred += reward
                total_ep += reward
                #  update Q values
                if not terminated:
                    self.Q[(S, A)] += self.alpha * (reward +
                                                    self.gamma * self.__greedy(S_) - self.Q[(S, A)])
                else:
                    self.Q[(S, A)] += self.alpha * (reward - self.Q[(S, A)])
                    break

                S = S_

            # Calculates rolling average
            rolling = np.roll(rolling, 1)
            rolling[0] = total_ep
            if i >= 99:
                y.append(np.mean(rolling))

            # accumulate average score over 100 episodes
            if i % 100 == 0 and i > 0:
                print("Episode Number: ",i," Average of previous 100: ", total_hundred/100)
                average_return.append(total_hundred / 100)
                total_hundred = 0.0

        # Plot the data
        x = list(range(100, len(y)+100))
        plt.plot(x, y)
        plt.xlabel("x - Episode Number")
        plt.ylabel("y - Average reward (per 100)")
        plt.show()

        # Save the Q values
        try:
            file = open("dict", "wb")
            pickle.dump(self.Q, file)
        except:
            print("Something went wrong")
        return average_return

    def play(self, n_episodes):
        self.Q = pickle.load(open("dict", "rb"))

        for _ in range(n_episodes):
            gym.make("LunarLander-v2", continuous=False)

            raw_state = self.env.reset()[0]
            state = discretize_state(raw_state)

            # for 1000 steps in the episode
            for _ in range(1000):
                self.env.render()

                #  choose action A from state S using epsilon greedy policy
                action = self.__epsilon_greedy(state)

                # take the action in the environment and observe the new state
                raw_new_state, reward, terminated, truncated, info = self.env.step(
                    action)
                game_callback(raw_state, raw_new_state, action, reward,
                              terminated, truncated, info)

                new_state = discretize_state(raw_new_state)

                if terminated:
                    break

                state = new_state

    def __greedy(self, state):
        return max([self.Q[(state, action)] for action in range(4)])

    def __epsilon_greedy(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(range(4))
        else:
            return np.argmax([self.Q[(state, action)] for action in range(4)])
