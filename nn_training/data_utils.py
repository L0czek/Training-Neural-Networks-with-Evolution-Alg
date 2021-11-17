import abc
import typing as t

import numpy as np


class IDataGenerator(abc.ABC):
    def __init__(self, seed):
        pass

    @abc.abstractclassmethod
    def __next__(self) -> np.ndarray:
        pass
