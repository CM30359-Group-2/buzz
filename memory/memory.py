from agents.transition import Transition

class Memory(object):
    def add(self, t: Transition):
        raise NotImplementedError()

    def sample()->"list[Transition]":
        raise NotImplementedError()