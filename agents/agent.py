from gym import Env
from memory.memory import Memory
from agents.transition import Transition

class Agent:
    def __init__(self, action_space, state_space, memory: Memory):
        self.action_space = action_space
        self.state_space = state_space
        self.memory = memory

    def act(self, state):
        pass

    def train(self, env: Env) -> "list[float]":
        pass

    def remember(self, transition: Transition):
        self.memory.add(transition)
    
    def save_model(self, episode: int):
        pass

    def load(self):
        pass