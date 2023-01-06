from agents.transition import Transition
from memory.memory import Memory


class QTable(Memory):
    def __init__(self):
        self.q_table = {}
    
    def update(self, transition: Transition, q_value: float):
        self.q_table.update({(transition.state, transition.action): q_value})

    def add(self, transition: Transition):
        self.q_table.update({(transition.state, transition.action): 0.0})
    
    def recall(self, state, action=None):
        if action is None:
            # Get all the values for the given state
            return [self.q_table.get((state, action)) for action in range(0, 4)]
        else:
            return self.q_table.get((state, action))

        