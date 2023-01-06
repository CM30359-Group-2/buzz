from agents.transition import Transition

class Memory(object):
    def add(t: Transition):
        raise NotImplementedError()

    def sample()->"list[Transition]":
        raise NotImplementedError()