import abc
import typing as t
import random

import numpy as np


class IDataGenerator(abc.ABC):
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def random(self) -> float:
        return self.rng.random()


    @abc.abstractmethod
    def __next__(self) -> np.ndarray:
        pass

    def __iter__(self):
        return self


class UniformDistribution(IDataGenerator):
    def __init__(self, seed, start, end):
        super().__init__(seed)
        self.start = start
        self.end = end

    def __next__(self) -> np.ndarray:
        return np.array([self.random() * (self.end - self.start) + self.start])

class RandintGenerator(IDataGenerator):
    def __init__(self, seed):
        super().__init__(seed)

    def __next__(self) -> np.ndarray:
        return np.array([ self.rng.randint(0, 1) for _ in range(2) ])


