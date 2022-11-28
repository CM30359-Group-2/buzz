from keras import Sequential
from keras.layers import Dense
import numpy as np
import keras.backend as K
import tensorflow as tf
from keras.regularizers import l2


def masked_huber_loss(mask_value, clip_delta):
    def f(y_true, y_pred):
        error = y_true - y_pred
        cond = K.abs(error) < clip_delta
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = 0.5 * K.square(mask_true * (y_true - y_pred))
        linear_loss = mask_true * \
            (clip_delta * K.abs(error) - 0.5 * (clip_delta ** 2))
        huber_loss = tf.where(cond, masked_squared_error, linear_loss)
        return K.sum(huber_loss) / K.sum(mask_true)
    f.__name__ = 'masked_huber_loss'
    return f


class QNetwork():
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, model=None):
        if model is not None:
            self.model = model
        else:
            self.model = Sequential([
                # Input layer
                Dense(64, activation='relu', input_shape=(
                    state_size,), kernel_regularizer=l2(0.01)),
                Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
                Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
                Dense(action_size, activation='linear',
                      kernel_regularizer=l2(0.01))
            ])
            self.model.compile(optimizer='nadam',
                               loss=masked_huber_loss(0.0, 1.0))

    def get_q_values(self, state):
        input = state[np.newaxis, ...]
        return self.model.predict(input)[0]

    def get_multiple_q_values(self, states):
        return self.model.predict(states)
