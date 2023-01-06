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