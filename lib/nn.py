from __future__ import annotations
import numpy as np
import abc

class INeuralNetwork(abc.ABC):
    def __init__(self, in_channels: int, n_hidden_neurons: int):
        self.in_channles = in_channels
        self.n_hidden_neurons = n_hidden_neurons

    @abc.abstractclassmethod
    def get_weights(self) -> np.ndarray:
        pass

    @abc.abstractclassmethod
    def update_weights(self, data: np.ndarray) -> None:
        pass

    @abc.abstractclassmethod
    def predict(self, x: np.ndarray) -> float:
        pass

class EvolutionAlgNeuralNetwork(INeuralNetwork):

    def __init__(self, in_channels: int, n_hidden_neurons: int):
        super().__init__(in_channels, n_hidden_neurons)
        self.hidden_layer = np.random.normal(size=(self.n_hidden_neurons, self.in_channles + 1))
        self.output_layer = np.random.normal(size=(1, self.n_hidden_neurons + 1))

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.hidden_layer.flatten(), self.output_layer.flatten()])

    def update_weights(self, weights: np.ndarray) -> None:
        pivot = (self.n_hidden_neurons) * (self.in_channles + 1)
        a, b = weights[:pivot], weights[pivot:]
        self.hidden_layer = a.reshape((self.n_hidden_neurons, self.in_channles + 1))
        self.output_layer = b.reshape((1, self.n_hidden_neurons + 1))

    def predict(self, x: np.ndarray) -> float:
        x = np.concatenate([x, [1]])
        hidden_out = np.dot(self.hidden_layer, x) # TODO: add sigmoid
        hidden_out = np.concatenate([hidden_out, [1]])
        output = np.dot(self.output_layer, hidden_out)
        return output

class SGDNeuralNetwork(INeuralNetwork):
    pass

