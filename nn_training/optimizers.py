from __future__ import annotations

import abc
import copy
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
            return len(self.individuals)

    def __init__(
        self,
        mu_value: int,
        lambda_value: int,
        optimized_func: t.Callable[[float], float],
        dataset: utils.IDataGenerator,
    ):
        assert lambda_value >= mu_value

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
        mutation_tau: float,
        mutation_tau_prime: float,
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

        number_of_weights = len(
            n_net.EvolutionAlgNeuralNetwork(
                in_channels, n_hidden_neurons, out_channels
            ).get_weights()
        )

        start = time.time()

        curr_population = MuLambdaEvolutionStrategy.Population(
            individuals=[
                n_net.EvolutionAlgNeuralNetwork(in_channels, n_hidden_neurons, out_channels)
                for _ in range(self.mu_value)
            ],
            sigmas=[[1 for _ in range(number_of_weights)] for _ in range(self.mu_value)],
        )

        losses = self._assess_population(curr_population)
        alg_trace.losses_per_epoch.append(losses)

        min_loss = min(losses)
        alg_trace.best_individual = curr_population.individuals[losses.index(min_loss)]
        alg_trace.best_individual_loss = min_loss
        print(f"Epoch 0 loss => {min_loss:.4f}")

        iteration = 1
        while iteration <= n_iters and min(losses) > best_loss_treshold:
            new_population = self._select4reproduce(curr_population, losses)

            new_population = self._mutation(
                new_population, tau=mutation_tau, tau_prime=mutation_tau_prime
            )
            new_population = self._crossover(new_population)

            losses = self._assess_population(new_population)

            # Returns population sorted by losses in ascending order.
            curr_population, losses = self._select_mu_best(new_population, losses)

            alg_trace.losses_per_epoch.append(losses)
            print(f"Epoch {iteration} loss => {losses[0]:.4f}")

            if losses[0] < alg_trace.best_individual_loss:
                alg_trace.best_individual = curr_population.individuals[0]
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
            loss = 0.0
            for x in self.dataset:
                y_pred = individual.predict([x])
                y_true = self.optimized_func(x)

                loss += ev.MSE(y_pred, y_true)

            losses.append(loss / len(self.dataset))

        return losses

    def _select4reproduce(
        self, population: MuLambdaEvolutionStrategy.Population, losses: t.List[float]
    ) -> MuLambdaEvolutionStrategy.Population:
        losses = np.array(losses)
        losses = losses.max() - losses
        selection_probabilities = losses / losses.sum()
        chosen_indices = np.random.choice(
            range(len(population)), size=self.lambda_value, p=selection_probabilities
        )

        individuals = [copy.deepcopy(population.individuals[index]) for index in chosen_indices]
        sigmas = [population.sigmas[index] for index in chosen_indices]
        return MuLambdaEvolutionStrategy.Population(individuals=individuals, sigmas=sigmas)

    def _crossover(
        self, population: MuLambdaEvolutionStrategy.Population
    ) -> MuLambdaEvolutionStrategy.Population:
        individuals_as_vector = [net.get_weights() for net in population.individuals]
        sigmas = [sigma for sigma in population.sigmas]

        crossovers = random.sample(range(len(individuals_as_vector)), len(individuals_as_vector))
        crossover_pairs = [
            (crossovers[ind_idx * 2], crossovers[ind_idx * 2 + 1])
            for ind_idx in range(len(crossovers) // 2)
        ]

        new_weights = []
        new_sigmas = []

        for f_individual, s_individual in crossover_pairs:
            f_weights = individuals_as_vector[f_individual]
            s_weights = individuals_as_vector[s_individual]

            crossovered_weights = (np.array(f_weights) + np.array(s_weights)) / 2.0

            f_sigma = sigmas[f_individual]
            s_sigma = sigmas[s_individual]

            crossovered_sigma = (np.array(f_sigma) + np.array(s_sigma)) / 2.0

            new_weights.append(crossovered_weights)
            new_sigmas.append(crossovered_sigma)

        # Add last individual to list
        if len(crossovers) // 2 != 0:
            new_weights.append(individuals_as_vector[crossovers[-1]])
            new_sigmas.append(sigmas[crossovers[-1]])

        new_individuals = []
        for net, weights in zip(population.individuals, new_weights):
            net.update_weights(weights)
            new_individuals.append(copy.deepcopy(net))

        return MuLambdaEvolutionStrategy.Population(individuals=new_individuals, sigmas=new_sigmas)

    def _mutation(
        self, population: MuLambdaEvolutionStrategy.Population, tau: float, tau_prime: float
    ) -> MuLambdaEvolutionStrategy.Population:
        individuals_as_vector = [net.get_weights() for net in population.individuals]

        random_values = list(np.random.normal(size=len(population.sigmas)))
        sigmas = [
            [
                float(sigma * np.exp(tau_prime * ind_random + tau * np.random.normal(size=1)))
                for sigma in individual_sigmas
            ]
            for individual_sigmas, ind_random in zip(population.sigmas, random_values)
        ]

        new_weights = [
            [
                float(weight + sigma * np.random.normal(size=1))
                for weight, sigma in zip(individual_weights, individual_sigmas)
            ]
            for individual_weights, individual_sigmas in zip(individuals_as_vector, sigmas)
        ]

        new_individuals = []
        for net, weights in zip(population.individuals, new_weights):
            net.update_weights(np.array(weights))
            new_individuals.append(copy.deepcopy(net))

        return MuLambdaEvolutionStrategy.Population(individuals=new_individuals, sigmas=sigmas)

    def _select_mu_best(
        self,
        population: MuLambdaEvolutionStrategy.Population,
        losses: t.List[float],
    ) -> t.Tuple[MuLambdaEvolutionStrategy.Population, t.List[float]]:
        individuals, sigmas = population.individuals, population.sigmas
        zipped4sort = zip(losses, individuals, sigmas)
        zipped4sort = sorted(zipped4sort, key=lambda x: x[0])

        sorted_losses, sorted_individuals, sorted_sigmas = [], [], []
        for loss, individual, sigma in zipped4sort:
            sorted_losses.append(loss)
            sorted_individuals.append(individual)
            sorted_sigmas.append(sigma)

        return (
            MuLambdaEvolutionStrategy.Population(
                individuals=sorted_individuals[: self.mu_value],
                sigmas=sorted_sigmas[: self.mu_value],
            ),
            sorted_losses[: self.mu_value],
        )
