import random


class Transition:
    def __init__(self, state, action, reward, next_state, done):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done

    def __repr__(self):
        return self.__str__()

    def __str__(self) -> str:
        return f"Transition(state={self.state}, action={self.action}, reward={self.reward}, next_state={self.next_state}, done={self.done})"


class ReplayBuffer:
    current_index = 0

    def __init__(self, buffer_size, batch_size, seed):
        self.size = buffer_size
        self.memory = []
        self.batch_size = batch_size
        self.seed = random.seed(seed)

    def add(self, transition: Transition):
        if len(self.memory) < self.size:
            self.memory.append(transition)
        else:
            self.memory[self.current_index] = transition
            self.__increment_current_index()

    def sample(self) -> "list[Transition]":
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)

    def __increment_current_index(self):
        self.current_index = (self.current_index + 1) % self.size
