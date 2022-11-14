from keras import Sequential
from keras.layers import Dense, Input

class QNetwork():
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):

        self.model = Sequential([
            Input(shape=(state_size,)),
            Dense(64, activation='relu', input_dim=state_size),
            Dense(64, activation='relu'),
            Dense(action_size, activation='softmax')
        ])
        self.model.compile(optimizer='nadam', loss='categorical_crossentropy', metrics=['accuracy'])


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.model(state)
        return x