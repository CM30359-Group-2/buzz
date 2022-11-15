import numpy as np


def discretize_state(s, ep):
    state_ = s
    # print(ep, "State in discretize:", state_)
    # print(type(state_))
    discrete_state = (min(2, max(-2, int(state_[0] / 0.05))), \
                        min(2, max(-2, int(state_[1] / 0.1))), \
                        min(2, max(-2, int(state_[2] / 0.1))), \
                        min(2, max(-2, int(state_[3] / 0.1))), \
                        min(2, max(-2, int(state_[4] / 0.1))), \
                        min(2, max(-2, int(state_[5] / 0.1))), \
                        int(state_[6]), \
                        int(state_[7]))
    # print("Discrete state:  ",discrete_state)
    # print(type(discrete_state))
    return discrete_state

class QLearning:
    def __init__(self, env, alpha = 0.5, gamma = 0.9, epsilon = 1.0):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        self.Q = {}
        #num actions = 4

        for act in range(0,4):
            for a in range(-2,3):
                for b in range(-2,3):
                    for c in range(-2,3):
                        for d in range(-2,3):
                            for e in range(-2,3):
                                for f in range(-2,3):
                                    for g in range(0,2):
                                        for h in range(0,2):
                                            temp = {((a,b,c,d,e,f,g,h),act):0}
                                            self.Q.update(temp)
        
        # print(self.Q)


       

    def train(self):
        for i in range(100000):

            S = discretize_state(self.env.reset()[0], i)
            #S = self.env.reset()

            while True:

                
                self.env.render()

                # choose action A from state S using epsilon greedy policy
                A = self.__epsilon_greedy(S)
                # print(A)
                # print("Action^\n\n")
                # take action A, take immediate reward R and move to state S_
                # S_, reward, terminated, _, _ = self.env.step(A)
                new_S, reward, terminated, _, _ = self.env.step(A)
                # print(new_S)
                # print("New State^\n\n")
                S_ = discretize_state(new_S, 2)
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
            return np.random.choice(range(4))
        else:
            return self.__greedy(state)
