import random

import numpy as np
from agents.transition import Transition
from memory.memory import Memory
from memory.partitioned_ring_buffer import PartitionedRingBuffer
from util.segment_tree import MinSegmentTree, SumSegmentTree


class PartitionedMemory(Memory):
    def __init__(self, limit, batch_size, seed, alpha=.4, beta=.6, **kwargs):
        super(PartitionedMemory, self).__init__(**kwargs)
        random.seed(seed)

        self.batch_size = batch_size
        self.limit = limit
        self.transitions = PartitionedRingBuffer(limit)
        assert alpha >= 0

        self.alpha = alpha
        self.beta = beta

        tree_capacity = 1
        while tree_capacity < self.limit:
            tree_capacity *= 2
        
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        self.max_priority = 1.0

    def load(self, load_data):
        self.transitions.load(load_data)
        self.permanent_idx = self.transitions.permanent_index
        
        self.next_idx = 0

        for idx in range(self.permanent_idx):
            self.sum_tree[idx] = self.max_priority ** self.alpha
            self.min_tree[idx] = self.max_priority ** self.alpha
    
    def add(self, transition):
        self.transitions.append(transition)

        self.sum_tree[self.next_idx + self.permanent_idx] = (self.max_priority ** self.alpha)
        self.min_tree[self.next_idx + self.permanent_idx] = (self.max_priority ** self.alpha)

        self.next_idx = (self.next_idx + 1) % (self.limit - self.permanent_idx)

    def sample_proportional(self):
        indices = list()

        for _ in range(self.batch_size):
            mass = random.random() * self.sum_tree.sum(0, self.limit - 1)
            idx = self.sum_tree.find_prefixsum_idx(mass)
            indices.append(idx)
        
        return indices

    def sample(self) -> "tuple[list[Transition], list[float]]":
        importance_weights = list()

        # The lowest-priority experience will have maximum importance sampling weight
        prob_min = self.min_tree.min() / self.sum_tree.sum()
        max_importance_weight = (prob_min * len(self.transitions)) ** (-self.beta)
        
        transitions = list()
        indices = self.sample_proportional()

        for idx in indices:
            transitions.append(self.transitions[idx])

            prob = self.sum_tree[idx] / self.sum_tree.sum()
            importance_weight = (prob * len(self.transitions)) ** (-self.beta)
            importance_weights.append(importance_weight / max_importance_weight)
        
        return transitions, np.array(importance_weights), indices


    def update_priorities(self, indices, priorities):
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < self.limit
            
            if idx < self.permanent_idx:
                priority = (priority ** self.alpha) + 0.999
            else:
                priority **= self.alpha
            
            self.sum_tree[idx] = priority
            self.min_tree[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self):
        return len(self.transitions)