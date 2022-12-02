import random
import numpy as np
from buffer import Transition


class PrioritisedReplayBuffer:

    def __init__(self, capacity, batch_size, prioritisation_amount):
        """
        Construct a new replay buffer.

        :param capacity: The maximum number of transitions that can be stored in the buffer.
        :param prioritisation_amount: The amount of prioritisation to use. 0 is uniform prioritisation, 1 is full prioritisation.
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.prioritisation_amount = prioritisation_amount
        self.priority_sum = [0 for _ in range(2 * self.capacity)]
        self.priority_min = [float('inf') for _ in range(2 * self.capacity)]
        self.max_priority = 1
        self.data = [None for _ in range(self.capacity)]
        self.next_index = 0
        self.size = 0

    def add(self, transition: Transition):
        index = self.next_index  # get the available slot
        self.data[index] = transition  # add the transition to the slot

        # increment the next available slot
        self.next_index = (index + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)  # calculate the size

        # new sample get maxPriority
        priority_alpha = self.max_priority ** self.prioritisation_amount

        # update the two segment trees
        self._set_priority_min(index, priority_alpha)
        self._set_priority_sum(index, priority_alpha)

    def _set_priority_min(self, index, priority_alpha):
        index += self.capacity
        self.priority_min[index] = priority_alpha

        # traverse the tree up to the the root
        while index >= 2:
            index //= 2  # visit the parent node
            # value of the parent node is the minimum of the left and right node
            self.priority_min[index] = min(
                self.priority_min[index * 2], self.priority_min[index * 2 + 1])

    def _set_priority_sum(self, index, priority):
        index += self.capacity
        self.priority_sum[index] = priority
        while index >= 2:
            index //= 2  # Visit the parent node
            self.priority_sum = self.priority_sum[index *
                                                  2] + self.priority_sum[index * 2 + 1]

    def _sum(self):
        return self.priority_sum[1]  # root node is the sum of all the values

    def _min(self):
        return self.priority_min[1]  # root node is the min of all the values

    def find_prefix_sum_index(self, prefix_sum):
        index = 1
        while index < self.capacity:
            if self.priority_sum[index * 2] > prefix_sum:
                index *= 2
            else:
                prefix_sum -= self.priority_sum[index * 2]
                index = index * 2 + 1

        return index - self.capacity

    def sample(self,  beta) -> "list[Transition]":
        samples = {

            'weights': np.zeros(shape=self.batch_size, dtype=np.float32),
            'indexes': np.zeros(shape=self.batch_size, dtype=np.int32)

        }

        for jdx in range(self.batch_size):
            p = random.random() * self.find_prefix_sum_index(p)
            idx = self.find_prefix_sum_index(p)
            samples['indexes'][jdx] = idx

        minimum_probability = self._min() / self._sum()

        max_weight = (minimum_probability * self.size) ** (-beta)

        for jdx in range(self.batch_size):

            idx = samples['indexes'][jdx]

            probability = self.priority_sum[idx + self.capacity] / self._sum()

            weight = (probability * self.size) ** (-beta)

            samples['weights'][jdx] = weight / max_weight

        for k, v in self.data.items():

            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):

        for idx, priority in zip(indexes, priorities):

            self.max_priority = max(self.max_priority, priority)

            priority_alpha = priority ** self.prioritisation_amount

            self._set_priority_sum(idx, priority_alpha)
            self._set_priority_min(idx, priority_alpha)

    def is_full(self):
        return self.capacity == self.size
