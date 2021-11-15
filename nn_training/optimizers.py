import abc
import random
import time
import typing as t
from dataclasses import dataclass
from statistics import mean

import numpy as np

import nn_training.data_utils as utils
import nn_training.evaluation as ev
import nn_training.neural_nets as n_net


@dataclass
class History:
    losses_per_epoch: t.List[t.List[float]]
    experiment_time_in_seconds: float


class IOptimizer(abc.ABC):
    @abc.abstractclassmethod
    def optimize(self, **kwargs) -> t.Tuple[n_net.EvolutionAlgNeuralNetwork, History]:
        pass

    @abc.abstractclassmethod
    def best(self) -> n_net.INeuralNetwork:
        pass


class MuLambdaEvolutionStrategy(IOptimizer):
    def __init__(
        self,
        mu_value: int,
        lambda_value: int,
        optimized_func: t.Callable[[float], float],
        dataset: utils.IDataGenerator,
    ):
        assert mu_value >= lambda_value
        super().__init__()

        self.mu_value = mu_value
        self.lambda_value = lambda_value
        self.optimized_func = optimized_func
        self.dataset = dataset
        self.best_individual = None

    def optimize(
        self,
        in_channels: int,
        n_hidden_neurons: int,
        out_channels: int,
        n_iters: int = 100,
        best_loss_treshold: float = 1e-3,
    ) -> t.Tuple[n_net.EvolutionAlgNeuralNetwork, History]:
        alg_history = History(losses_per_epoch=[], experiment_time_in_seconds=0)

        start = time.time()

        curr_population = [
            n_net.EvolutionAlgNeuralNetwork(in_channels, n_hidden_neurons, out_channels)
            for _ in range(self.mu_value)
        ]

        losses = self._assess_population(curr_population)
        alg_history.losses_per_epoch.append(losses)

        iteration = 0
        while iteration < n_iters and max(losses) > best_loss_treshold:
            new_population = self._select4reproduce(curr_population, losses)

            new_population = self._crossover(new_population)
            new_population = self._mutation(new_population)

            losses = self._assess_population(new_population)
            curr_population, losses = self._select_mu_best(new_population, losses)

            alg_history.losses_per_epoch.append(losses)

        end = time.time()

        alg_history.experiment_time_in_seconds = end - start
        self.best_individual = (
            curr_population[0] if iteration != 0 else curr_population[losses.index(min(losses))]
        )

        return self.best_individual, alg_history

    def best(self) -> n_net.INeuralNetwork:
        return self.best_individual

    def _assess_population(
        self, population: t.List[n_net.EvolutionAlgNeuralNetwork]
    ) -> t.List[float]:
        losses = []

        for individual in population:
            for x in self.dataset:
                y_pred = individual.predict(x)
                y_true = self.optimized_func(x)

                losses.append(ev.MSE(y_pred, y_true))

        return losses

    def _select4reproduce(
        self, population: t.List[n_net.EvolutionAlgNeuralNetwork], losses: t.List[float]
    ) -> t.List[n_net.EvolutionAlgNeuralNetwork]:
        # replace with something better
        return random.choices(population, k=self.lambda_value)

    def _mutation(
        self, population: t.List[n_net.EvolutionAlgNeuralNetwork]
    ) -> t.List[n_net.EvolutionAlgNeuralNetwork]:
        population_as_vector = [net.get_weights() for net in population]

        # Perform mutation

        return [net.update_weights(vector) for net, vector in zip(population, population_as_vector)]

    def _crossover(
        self, population: t.List[n_net.EvolutionAlgNeuralNetwork]
    ) -> t.List[n_net.EvolutionAlgNeuralNetwork]:
        population_as_vector = [net.get_weights() for net in population]

        # Perform crossover

        return [net.update_weights(vector) for net, vector in zip(population, population_as_vector)]

    def _select_mu_best(
        self,
        population: t.List[n_net.EvolutionAlgNeuralNetwork],
        losses: t.List[float],
    ) -> t.List[n_net.EvolutionAlgNeuralNetwork]:
        zipped4sort = zip(losses, population)
        zipped4sort = sorted(zipped4sort)

        sorted_population = [individual for _, individual in zipped4sort]
        sorted_losses = [loss for loss, _ in zipped4sort]

        return sorted_population[: self.mu_value], sorted_losses[: self.mu_value]
