import random
from agents.memories.memory import Memory

from agents.transition import Transition


class ReplayBuffer(Memory):
    current_index = 0

    def __init__(self, buffer_size, batch_size, seed):
        self.size = buffer_size
        self.memory = []
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, t: Transition):
        if len(self.memory) < self.size:
            self.memory.append(t)
        else:
            self.memory[self.current_index] = t
            self.__increment_current_index()

    def sample(self) -> "list[Transition]":
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

    def __increment_current_index(self):
        self.current_index = (self.current_index + 1) % self.size
