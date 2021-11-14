import abc
from dataclasses import dataclass
import typing as t

import numpy as np

import nn_training.neural_nets as neural_nets


@dataclass
class History:
    best_fitness_per_epoch: t.List[float]
    experiment_time: float


class IOptimizer(abc.ABC):
    @abc.abstractclassmethod
    def optimize(self, **kwargs) -> History:
        pass

    @abc.abstractclassmethod
    def best(self) -> neural_nets.INeuralNetwork:
        pass
