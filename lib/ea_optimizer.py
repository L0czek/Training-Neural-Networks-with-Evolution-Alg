from __future__ import annotations
import numpy as np
import abc
from .nn import INeuralNetwork
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class History:
    best_fitness : List[float]
    time : float

class IOptimizer(abc.ABC):

    @abc.abstractclassmethod
    def optimize(self, **kwargs) -> History:
        pass

    @abc.abstractclassmethod
    def best(self) -> INeuralNetwork:
        pass


class IDataGenerator(abc.ABC):
    def __init__(self, seed):
        pass

    @abc.abstractclassmethod
    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        pass
