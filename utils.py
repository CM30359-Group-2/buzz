import random
import shutil
import uuid
import numpy as np
from buffer import Transition
from keras import models

from q_network import masked_huber_loss

def calculate_target_values(agent, target_agent, state_transitions: 'list[Transition]', discount_factor):
    new_states = np.array([t.next_state for t in state_transitions])

    q_values = agent.get_multiple_q_values(new_states)
    target_q_values = target_agent.get_multiple_q_values(new_states)

    targets = []
    for index, state_transition in enumerate(state_transitions):
        best_action = select_best_action(q_values[index])
        best_action_q_value = target_q_values[index][best_action]

        if state_transition.done:
            target_value = state_transition.reward
        else:
            target_value = state_transition.reward + discount_factor * best_action_q_value

        target_vector = [0] * 4
        target_vector[state_transition.action] = target_value
        targets.append(target_vector)

    return np.array(targets)


def train_model(model, states, targets):
    model.fit(states, targets, epochs=1, batch_size=len(targets), verbose=0)


def copy_model(model):
    backup_file = 'backup_'+str(uuid.uuid4())
    model.save(backup_file)
    new_model = models.load_model(backup_file, custom_objects={
                                  'masked_huber_loss': masked_huber_loss(0.0, 1.0)})
    shutil.rmtree(backup_file)
    return new_model


def select_action_epsilon_greedy(q_values, epsilon):
    random_value = random.uniform(0, 1)
    if random_value < epsilon:
        return random.randint(0, len(q_values) - 1)
    else:
        return np.argmax(q_values)


def select_best_action(q_values):
    return np.argmax(q_values)