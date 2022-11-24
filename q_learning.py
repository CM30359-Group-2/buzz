import numpy as np
import time
import gym
import json



def discretize_state(s):
    state_ = s
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
    def __init__(self, env, alpha = 0.1, gamma = 0.99, epsilon = 1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}
        
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
        print(len(self.Q))


    def train(self):
        for i in range(10500):
            print(i)

            #Decay epsilon so later on it doesn't search as much
            self.epsilon = self.epsilon * 0.996

            #Start redering at 10000            
            if i == 10000:
                self.env = gym.make("LunarLander-v2", render_mode = "human", continuous=False)

            #Initial state
            S = discretize_state(self.env.reset()[0])
            timeout = time.time() + 60

            #Until the env terminates or 60 seconds pass
            while True:
                if time.time() > timeout:
                    break
                self.env.render()
                # choose action A from state S using epsilon greedy policy
                A = self.__epsilon_greedy(S, i)
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
        
        with open('qstates.txt', 'w') as f:
            f.write(json.dumps(self.Q))

    #Greedy Checked and is fine
    def __greedy(self, state):
        return np.argmax([self.Q[(state, action)] for action in range(4)])

    #Eps Greedy checked and is fine
    def __epsilon_greedy(self, state, eps):
        if np.random.random() < self.epsilon:
            return np.random.choice(range(4))
        else:
            return self.__greedy(state)
