from gym import Env
from memory.memory import Memory
from agents.transition import Transition
from typing import Type

class Agent:
    def __init__(self, action_space, state_space, memory: Type[Memory]):
        self.action_space = action_space
        self.state_space = state_space
        self.memory = memory

    def choose_action(self, state):
        raise NotImplementedError()

    def train(self, env: Env, episodes: int, checkpoint: bool, render: bool) -> "list[float]":
        raise NotImplementedError()

    def remember(self, transition: Transition):
        self.memory.add(transition)
    
    def play(env: Env, episodes: int, checkpoint_path: str, render: bool) -> "list[float]":
        raise NotImplementedError()

    def save_checkpoint(self, episode: int):
        raise NotImplementedError()

    def load_checkpoint(self, checkpoint_path: str):
        raise NotImplementedError()
