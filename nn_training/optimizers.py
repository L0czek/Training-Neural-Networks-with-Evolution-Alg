from __future__ import annotations

import abc
import random
import time
import typing as t
from dataclasses import dataclass

import numpy as np

import nn_training.data_utils as utils
import nn_training.evaluation as ev
import nn_training.neural_nets as n_net


@dataclass
class Experiment:
    name: str
    losses_per_epoch: t.List[t.List[float]]
    experiment_time_in_seconds: float
    best_individual: n_net.EvolutionAlgNeuralNetwork
    best_individual_loss: float
    best_individual_iteration: int


class IOptimizer(abc.ABC):
    @abc.abstractclassmethod
    def optimize(self, **kwargs) -> t.Tuple[n_net.EvolutionAlgNeuralNetwork, Experiment]:
        pass

    @abc.abstractclassmethod
    def best(self) -> n_net.INeuralNetwork:
        pass


class MuLambdaEvolutionStrategy(IOptimizer):
    @dataclass
    class Population:
        individuals: t.List[n_net.EvolutionAlgNeuralNetwork]
        sigmas: t.List[float]

        def __len__(self) -> int:
            return len(self.individual)

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
        experiment_name: str,
        in_channels: int,
        n_hidden_neurons: int,
        out_channels: int,
        n_iters: int = 100,
        best_loss_treshold: float = 1e-3,
    ) -> t.Tuple[n_net.EvolutionAlgNeuralNetwork, Experiment]:
        alg_trace = Experiment(
            name=experiment_name,
            losses_per_epoch=[],
            experiment_time_in_seconds=0,
            best_individual=None,
            best_individual_loss=float("inf"),
            best_individual_iteration=0,
        )

        start = time.time()

        curr_population = MuLambdaEvolutionStrategy.Population(
            individuals=[
                n_net.EvolutionAlgNeuralNetwork(in_channels, n_hidden_neurons, out_channels)
                for _ in range(self.mu_value)
            ],
            sigmas=[1 for _ in range(self.mu_value)],
        )

        losses = self._assess_population(curr_population)
        alg_trace.losses_per_epoch.append(losses)

        min_loss = min(losses)
        alg_trace.best_individual = curr_population[losses.index(min_loss)]
        alg_trace.best_individual_loss = min_loss

        iteration = 1
        while iteration <= n_iters and min(losses) < best_loss_treshold:
            new_population = self._select4reproduce(curr_population, losses)

            new_population = self._crossover(new_population)
            new_population = self._mutation(new_population)

            losses = self._assess_population(new_population)

            # Returns population sorted by losses in ascending order.
            curr_population, losses = self._select_mu_best(new_population, losses)

            alg_trace.losses_per_epoch.append(losses)

            if losses[0] < alg_trace.best_individual_loss:
                alg_trace.best_individual = curr_population[0]
                alg_trace.best_individual_loss = losses[0]
                alg_trace.best_individual_iteration = iteration

            iteration += 1

        end = time.time()

        alg_trace.experiment_time_in_seconds = end - start
        self.best_individual = alg_trace.best_individual

        return self.best_individual, alg_trace

    def best(self) -> n_net.INeuralNetwork:
        return self.best_individual

    def _assess_population(self, population: MuLambdaEvolutionStrategy.Population) -> t.List[float]:
        losses = []

        for individual in population.individuals:
            for x in self.dataset:
                y_pred = individual.predict(x)
                y_true = self.optimized_func(x)

                losses.append(ev.MSE(y_pred, y_true))

        return losses

    def _select4reproduce(
        self, population: MuLambdaEvolutionStrategy.Population, losses: t.List[float]
    ) -> MuLambdaEvolutionStrategy.Population:
        # replace with something better
        chosen_indices = np.random.randint(0, len(population), size=self.lambda_value)

        individuals = [population.individuals[index] for index in chosen_indices]
        sigmas = [population.sigmas[index] for index in chosen_indices]
        return MuLambdaEvolutionStrategy.Population(individuals=individuals, sigmas=sigmas)

    def _mutation(
        self, population: MuLambdaEvolutionStrategy.Population
    ) -> MuLambdaEvolutionStrategy.Population:
        individuals_as_vector = [net.get_weights() for net in population.individuals]
        sigmas = [net.get_weights() for net in population.sigmas]

        # Perform mutation

        new_individuals = [
            net.update_weights(vector)
            for net, vector in zip(population.individuals, individuals_as_vector)
        ]
        return MuLambdaEvolutionStrategy.Population(individuals=new_individuals, sigmas=sigmas)

    def _crossover(
        self, population: MuLambdaEvolutionStrategy.Population
    ) -> MuLambdaEvolutionStrategy.Population:
        individuals_as_vector = [net.get_weights() for net in population.individuals]
        sigmas = [net.get_weights() for net in population.sigmas]

        # Perform crossover

        new_individuals = [
            net.update_weights(vector)
            for net, vector in zip(population.individuals, individuals_as_vector)
        ]
        return MuLambdaEvolutionStrategy.Population(individuals=new_individuals, sigmas=sigmas)

    def _select_mu_best(
        self,
        population: MuLambdaEvolutionStrategy.Population,
        losses: t.List[float],
    ) -> t.Tuple[MuLambdaEvolutionStrategy.Population, t.List[float]]:
        individuals, sigmas = population.individuals, population.sigmas
        zipped4sort = zip(losses, individuals, sigmas)
        zipped4sort = sorted(zipped4sort)

        sorted_individuals = [individual for _, individual, _ in zipped4sort]
        sorted_sigmas = [sigma for _, _, sigma in zipped4sort]
        sorted_losses = [loss for loss, _, _ in zipped4sort]

        return (
            MuLambdaEvolutionStrategy.Population(
                individuals=sorted_individuals[: self.mu_value],
                sigmas=sorted_sigmas[: self.mu_value],
            ),
            sorted_losses[: self.mu_value],
        )
