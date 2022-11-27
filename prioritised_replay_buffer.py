import random
import numpy as np

class ReplayBuffer:

    def __init(self, capacity, prioritsationAmount):

        self.capacity = capacity # capacity of the binary tree
        self.prioritsationAmount = prioritsationAmount # prioritsationAmount dictates how much prioritsation is used, 0  = unifrom prioritsation
        
        self.prioritySum = [0 for idx in range(2 * self.capacity)]
        self.priorityMin = [float('inf') for idx in range(2 * self.capacity)]

        self.maxPriority = 1

        self.data = {
            'obs_t': np.zeros(shape=(capacity, 4, 84, 84), dtype=np.uint8),
            'obs_tp1': np.zeros(shape=(capacity, 4, 84, 84), dtype=np.uint8),
            'action': np.zeros(shape=capacity, dtype=np.int32),
            'rew': np.zeros(shape=capacity, dtype=np.float32),
            'terminated': np.zeros(shape=capacity, dtype=np.bool),
            'truncated': np.zeros(shape=capacity, dtype=np.bool),

        }

        self.nextIdx = 0 # index of the next empty slot

        self.size = 0 # sixe of the replay buffer

    def add(self, obs_t, obs_tp1, action, rew, terminated, truncated):

        idx = self.nextIdx # get the available slot

        # store data in the queue
        self.data['obs_t'][idx] = obs_t
        self.data['obs_tp1'][idx] = obs_tp1
        self.data['action'][idx] = action
        self.data['rew'][idx] = rew
        self.data['terminated'][idx] = terminated
        self.data['truncated'][idx] = truncated

        self.nextIdx = (idx + 1) % self.capacity # increment the next available slot

        self.size = min(self.capacity, self.size + 1) # calculate the size

        priorityAlpha = self.maxPriority ** self.prioritsationAmount # new sample get maxPriority

        # update the two segment trees
        self._setPriorityMin(idx, priorityAlpha)
        self._setPrioritySum(idx, priorityAlpha)

    def _setPriorityMin(self, idx, priorityAlpha):

        idx += self.capacity
        self.priorityMin[idx] = priorityAlpha

        #traverse the tree up to the the root
        while idx >= 2:

            idx //= 2 # idx of the parent node

            # value of the parent node is the minimum of the left and right node
            self.priorityMin[idx] = min(self.priorityMin[idx * 2], self.priorityMin[idx * 2 + 1])

    def _setPrioritySum(self, idx, priority):

        idx += self.capacity

        self.prioritySum[idx] = priority

        while idx >= 2:

            idx //= 2 # idx of the parent node

            self.prioritySum = self.prioritySum[idx * 2] + self.prioritySum[idx * 2 + 1]

    def _sum(self):

        return self.prioritySum[1] # root node is the sum of all the values

    def _min(self):

        return self.priorityMin[1] # root node is the min of all the values


    def findPrefixSumIdx(self, prefixSum):

        idx = 1
        while idx < self.capacity:

            if self.prioritySum[idx * 2] > prefixSum:

                idx *= 2

            else:

                prefixSum -= self.prioritySum[idx * 2]
                idx = idx * 2 + 1

        return idx - self.capacity

    def sample(self, batchSize, beta):

        samples = {

            'weights': np.zeros(shape=batchSize, dtype=np.float32),
            'indexes': np.zeros(shape=batchSize, dtype=np.int32)

        }

        for jdx in range(batchSize):
            p = random.random() * self.findPrefixSumIdx(p)
            idx = self.findPrefixSumIdx(p)
            samples['indexes'][jdx] = idx

        probMin = self._min() / self._sum()

        maxWeight = (probMin * self.size) ** (-beta)

        for jdx in range(batchSize):

            idx = samples['indexes'][i]

            prob = self.priority_sum[idx + self.capacity] / self._sum()

            weight = (prob * self.size) ** (-beta)

            samples['weights'][jdx] = weight / maxWeight

        for k, v in self.data.items():

            samples[k] = v[samples['indexes']]

        return samples

    def update_priorities(self, indexes, priorities):

        for idx, priority in zip(indexes, priorities):

            self.maxPriority = max(self.maxPriority, priority)

            priorityAlpha = priority ** self.prioritsationAmount

            self._setPrioritySum(idx, priorityAlpha)
            self._setPriorityMin(idx, priorityAlpha)

    def is_full(self):

        if self.capacity == self.size:

            return True

        else:

            return False