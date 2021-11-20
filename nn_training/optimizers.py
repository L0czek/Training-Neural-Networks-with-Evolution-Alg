from __future__ import annotations

import abc
import copy
import random
import time
import typing as t
import sys
from dataclasses import dataclass

import numpy as np
import torch

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

    def optimize(
        self,
        *,
        experiment_name: str,
        in_channels: int,
        n_hidden_neurons: int,
        out_channels: int,
        mutation_tau: float,
        mutation_tau_prime: float,
        n_iters: int = 100,
        probe_times: int = 1000,
        best_loss_treshold: float = 1e-3,
    ) -> Experiment:
        experiment = Experiment(
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

        losses = self._assess_population(curr_population, probe_times)
        experiment.losses_per_epoch.append(losses)

        min_loss = min(losses)
        experiment.best_individual = curr_population.individuals[losses.index(min_loss)]
        experiment.best_individual_loss = min_loss
        print(f"Epoch 0 loss => {min_loss:.4f}")

        iteration = 1
        while iteration <= n_iters and min(losses) > best_loss_treshold:
            new_population = self._select4reproduce(curr_population, losses)

            new_population = self._crossover(new_population)
            new_population = self._mutation(
                new_population, tau=mutation_tau, tau_prime=mutation_tau_prime
            )

            losses = self._assess_population(new_population, probe_times)

            # Returns population sorted by losses in ascending order.
            curr_population, losses = self._select_mu_best(new_population, losses)

            experiment.losses_per_epoch.append(losses)
            print(f"Epoch {iteration} loss => {losses[0]:.4f}")

            if losses[0] < experiment.best_individual_loss:
                experiment.best_individual = copy.deepcopy(curr_population.individuals[0])
                experiment.best_individual_loss = losses[0]
                experiment.best_individual_iteration = iteration

            iteration += 1

        end = time.time()

        experiment.experiment_time_in_seconds = end - start

        return experiment

    def _assess_population(
        self, population: MuLambdaEvolutionStrategy.Population, probe_times: int
    ) -> t.List[float]:
        losses = []

        for individual in population.individuals:
            loss = 0.0

            for _, x in zip(range(probe_times), self.dataset):
                y_pred = individual.predict(x)
                y_true = self.optimized_func(x)

                loss += ev.MSE(y_pred, y_true)

            losses.append(loss / probe_times)

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


class DiffEvolution(IOptimizer):
    def __init__(
        self,
        *,
        optimized_func: t.Callable[[float], float],
        dataset: utils.IDataGenerator,
        population_size: int = 1000,
    ):
        self.optimized_func = optimized_func
        self.dataset = dataset
        self.population_size = population_size

    def optimize(
        self,
        *,
        experiment_name: str,
        in_channels: int,
        n_hidden_neurons: int,
        out_channels: int,
        f_factor: float = 0.5,
        min_f_factor: float = 0.01,
        n_iters: int = 100,
        best_loss_treshold: float = 1e-3,
        probe_times: int = 1000,
        gamma: float = 1.0,
    ) -> Experiment:
        experiment = Experiment(
            name=experiment_name,
            losses_per_epoch=[],
            experiment_time_in_seconds=0,
            best_individual=None,
            best_individual_loss=float("inf"),
            best_individual_iteration=0,
        )

        start = time.time()

        population = [
            n_net.EvolutionAlgNeuralNetwork(in_channels, n_hidden_neurons, out_channels)
            for _ in range(self.population_size)
        ]

        for iteration in range(n_iters):
            for ind, current_specimen in enumerate(population):
                r = np.random.choice(population)
                d = np.random.choice(population)
                e = np.random.choice(population)

                mutated_specimen_weights = r.get_weights() + f_factor * (
                    e.get_weights() - d.get_weights()
                )
                new_specimen_wieghts = self._cross_specimen_weights_bin(
                    mutated_specimen_weights, r.get_weights()
                )
                new_specimen = n_net.EvolutionAlgNeuralNetwork(
                    in_channels, n_hidden_neurons, out_channels
                )
                new_specimen.update_weights(new_specimen_wieghts)

                population[ind] = self._tournament(current_specimen, new_specimen, probe_times)

            self._trace_loss(
                population, experiment, probe_times, iteration
            )
            f_factor = max(f_factor * gamma, min_f_factor)

            if experiment.best_individual_loss < best_loss_treshold:
                break

        stop = time.time()
        experiment.experiment_time_in_seconds = stop - start

        return experiment

    def _assess_specimen(
        self, specimen: n_net.EvolutionAlgNeuralNetwork, probe_times: int
    ) -> float:
        loss = np.zeros((probe_times))
        for i, x in zip(range(probe_times), self.dataset):
            loss[i] = ev.MSE(specimen.predict(x), self.optimized_func(x))
        return np.average(loss)

    def _assess_population(
        self, population: t.List[n_net.EvolutionAlgNeuralNetwork], probe_times: int
    ) -> t.List[float]:
        losses = np.zeros((len(population)))
        for ind, specimen in enumerate(population):
            losses[ind] = self._assess_specimen(specimen, probe_times)
        return losses

    def _cross_specimen_weights_bin(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        selector = np.random.randint(2, size=a.shape[0])
        inv_selector = np.ones(a.shape) - selector
        # This is just a "bit mask" to quickly take either value either from a or b
        return a * selector + b * inv_selector

    def _cross_specimen_weights_pivot(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        end = a.shape[0]
        pivot = np.random.randint(0, end)
        selector = np.concatenate([np.ones(pivot), np.zeros(end - pivot)])
        inv_selector = np.ones(end) - selector
        return a * selector + b * inv_selector

    def _tournament(
        self,
        a: n_net.EvolutionAlgNeuralNetwork,
        b: n_net.EvolutionAlgNeuralNetwork,
        probe_times: int,
    ) -> n_net.EvolutionAlgNeuralNetwork:
        loss_a = self._assess_specimen(a, probe_times)
        loss_b = self._assess_specimen(b, probe_times)

        if loss_a < loss_b:
            return a
        else:
            return b

    def _trace_loss(
        self,
        population: t.List[n_net.EvolutionAlgNeuralNetwork],
        experiment: Experiment,
        probe_times: int,
        epoch: int,
    ) -> None:
        loss = self._assess_population(population, probe_times)
        min_loss_index = np.argmin(loss)
        min_loss = loss[min_loss_index]
        experiment.best_individual_iteration = epoch

        experiment.losses_per_epoch.append(loss)

        if min_loss < experiment.best_individual_loss:
            experiment.best_individual_iteration = min_loss_index
            experiment.best_individual_loss = min_loss
            experiment.best_individual = copy.deepcopy(population[min_loss_index])

        sys.stdout.write(
            f"\rEpoch {epoch} loss = {experiment.best_individual_loss}"
        )
        sys.stdout.flush()

        return loss, min_loss


class GradientDescent(IOptimizer):
    def __init__(
        self,
        optimized_func: t.Callable[[float], float],
        dataset: utils.IDataGenerator,
    ):
        super().__init__()

        self.optimized_func = optimized_func
        self.dataset = dataset

    def optimize(
        self,
        *,
        experiment_name: str,
        in_channels: int,
        n_hidden_neurons: int,
        out_channels: int,
        lr: float,
        n_iters: int = 100,
        probe_times: int = 1000,
        best_loss_treshold: float = 1e-3,
    ) -> Experiment:
        experiment = Experiment(
            name=experiment_name,
            losses_per_epoch=[],
            experiment_time_in_seconds=0,
            best_individual=None,
            best_individual_loss=float("inf"),
            best_individual_iteration=0,
        )

        neural_net = n_net.GradientDescentNeuralNetwork(in_channels, n_hidden_neurons, out_channels)

        curr_loss = float("inf")
        iteration = 0
        while iteration <= n_iters and curr_loss > best_loss_treshold:
            curr_loss = self._epoch(neural_net, lr, probe_times)
            experiment.losses_per_epoch.append(curr_loss)

            print(f"Epoch {iteration} loss => {curr_loss}")

            if curr_loss < experiment.best_individual_loss:
                experiment.best_individual = copy.deepcopy(neural_net)
                experiment.best_individual_loss = curr_loss
                experiment.best_individual_iteration = iteration

            iteration += 1

        return experiment

    def _epoch(
        self,
        neural_net: n_net.GradientDescentNeuralNetwork,
        lr: float,
        probe_times: int,
    ) -> float:
        epoch_loss = 0.0
        for _, x in zip(range(probe_times), self.dataset):
            y_pred = neural_net.forward(torch.tensor(x, dtype=torch.float))
            y_true = self.optimized_func(x)

            loss = torch.nn.functional.mse_loss(y_pred.float(), torch.tensor(y_true).float())
            loss.backward()

            epoch_loss += loss.item()

            with torch.no_grad():
                for parameter in neural_net.parameters():
                    parameter -= lr * parameter.grad
                    parameter.grad.zero_()

        epoch_loss /= probe_times
        return epoch_loss
