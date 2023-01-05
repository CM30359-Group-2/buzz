from agents.transition import Transition


class PartitionedRingBuffer(object):
    """
    A ring buffer that partitions the buffer into multiple segments.
    One segment cannot be overwritten.
    """

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.length = 0
        self.data = [None for _ in range(max_size)]
        self.permanent_index = 0
        self.next_index = 0

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx) -> Transition:
        if idx < 0:
            raise KeyError("Negative indexing is not supported")
        return self.data[idx % self.max_size]

    def append(self, t: Transition):
        if self.length < self.max_size:
            self.length += 1
        self.data[(self.permanent_index + self.next_index )] = t
        self.next_index = (self.next_index + 1) % (self.max_size - self.permanent_index)

    def load(self, load_data: "list[Transition]"):
        assert len(load_data) < self.max_size, "Cannot load data larger than the buffer"
        for idx, data in enumerate(load_data):
            self.length += 1
            self.data[idx] = data
            self.permanent_index += 1
