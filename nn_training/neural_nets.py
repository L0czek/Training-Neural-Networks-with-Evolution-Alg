import abc

import numpy as np


class INeuralNetwork(abc.ABC):
    def __init__(self, in_channels: int, n_hidden_neurons: int, out_channels: int):
        self.in_channels = in_channels
        self.n_hidden_neurons = n_hidden_neurons
        self.out_channels = out_channels

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
    def __init__(self, in_channels: int, n_hidden_neurons: int, out_channels: int):
        super().__init__(in_channels, n_hidden_neurons, out_channels)

        self.hidden_layer = np.random.normal(size=(self.n_hidden_neurons, self.in_channels + 1))
        self.output_layer = np.random.normal(size=(self.out_channels, self.n_hidden_neurons + 1))

    def get_weights(self) -> np.ndarray:
        return np.concatenate([self.hidden_layer.flatten(), self.output_layer.flatten()])

    def update_weights(self, weights: np.ndarray) -> None:
        pivot = (self.n_hidden_neurons) * (self.in_channels + 1)
        in2hidden, hidden2out = weights[:pivot], weights[pivot:]

        self.hidden_layer = in2hidden.reshape((self.n_hidden_neurons, self.in_channels + 1))
        self.output_layer = hidden2out.reshape((1, self.n_hidden_neurons + 1))

    def predict(self, x: np.ndarray) -> float:
        x = np.concatenate([x, [1]])

        hidden_out = self._leaky_relu(np.dot(self.hidden_layer, x))
        hidden_out = np.concatenate([hidden_out, [1]])

        output = np.dot(self.output_layer, hidden_out)

        return output

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _leaky_relu(self, x: np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, x * 0.01)


class SGDNeuralNetwork(INeuralNetwork):
    pass
