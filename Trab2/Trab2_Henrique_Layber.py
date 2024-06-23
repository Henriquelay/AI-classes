"""No caso do SA, uma iteração corresponde a avaliação de todas soluções em uma mesma temperatura. o total de avaliações em uma mesma temperatura, devem ser fixados em 100. Também deve ser limitado a 1000 o número máximo de iterações dos algoritmos."""

# pylint: disable=redefined-outer-name

from math import exp
from random import random


import numpy as np


from Dino.dinoAIParallel import (
    playGame,
    KeyClassifier,
    LargeCactus,
    SmallCactus,
    Bird,
    Obstacle,
)

type InputWeights = list[list[float]]  # 7x4 matrix of input x hidden weights
type HiddenWeights = list[list[float]]  # 4x1 matrix of hidden x output weights
type Biases = tuple[float, float]  # layer biases

type State = tuple[InputWeights, HiddenWeights, Biases]
"""Weight, biases of the neural network."""


def normalize_obstable(obType: Obstacle) -> float:
    if isinstance(obType, LargeCactus):
        return 0.0
    if isinstance(obType, SmallCactus):
        return 0.5
    if isinstance(obType, Bird):
        return 1.0
    if isinstance(obType, float):
        return obType
    if isinstance(obType, int):
        return float(obType)
    raise ValueError(f"Unknown obstacle type: {obType}")


class NeuralNetwork(KeyClassifier):
    def __init__(self, state: State, activation_threshold=0.55, _print=False):
        self.input_weights = state[0]
        self.hidden_weights = state[1]
        self.input_bias, self.hidden_bias = state[2]
        self.activation_threshold = activation_threshold
        self._print = _print

    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x) -> float:
        return 1 / (1 + np.exp(-x))

    def feedforward(self, inputs: list[float]) -> float:
        """Weights is what will be searched for by the simulated annealing algorithm."""
        hidden_layer = self.relu(np.dot(inputs, self.input_weights) + self.input_bias)
        output_layer = self.relu(
            np.dot(hidden_layer, self.hidden_weights) + self.hidden_bias
        )
        return self.sigmoid(output_layer)

    def keySelector(
        self,
        distance,
        obHeight,
        speed,
        obType,
        nextObDistance,
        nextObHeight,
        nextObType,
    ):
        obType = normalize_obstable(obType)
        nextObType = normalize_obstable(nextObType)

        prediction = self.feedforward(
            np.array(
                [
                    distance,
                    obHeight,
                    speed,
                    obType,
                    nextObDistance,
                    nextObHeight,
                    nextObType,
                ],
                dtype=np.float32,
            )
        )

        if self._print:
            print(prediction, end="")
        if prediction >= self.activation_threshold:
            # if self._print:
            # print("🔼")
            return "K_UP"
        else:
            if self._print:
                print("🔽")
            return "K_DOWN"


class SimulatedAnnealing:
    """Simulated Annealing algorithm."""

    def __init__(
        self,
        initial_state: State,
        temperature=100,
        max_iter=1000,
        cooling_rate=0.003,
        states_per_iter=100,
    ):
        self.initial_state = initial_state
        self.temperature = temperature
        self.initial_temperature = temperature
        self.max_iter = max_iter
        self.cooling_rate = cooling_rate
        self.states_per_iter = states_per_iter

    def metropolis(self, energy, new_energy):
        """Probability of accepting a worse solution."""
        print(f"{energy=:.4f} {new_energy=:.4f} {self.temperature=:.2f}", end="")
        metropolis = np.exp(-(new_energy - energy) / self.temperature)
        print(f" {metropolis=}")
        return metropolis

    def derive_state(self, state: State) -> State:
        def perturbation() -> float:
            return random() * (self.initial_temperature / self.temperature)

        input_weights = [[w * perturbation() for w in weights] for weights in state[0]]
        hidden_weights = [[w * perturbation() for w in weights] for weights in state[1]]
        biases = tuple(w * perturbation() for w in state[2])

        return (input_weights, hidden_weights, biases)

    def anneal(self):
        """
        Anneals the state.

        Returns the best state found.

        The algorithm will run for `max_iter` iterations, each iteration
        will evaluate 100 possible states at the current temperature.

        A lower energy is better.

        Best overall is not kept to make it more resilient to local minima
        (outliers and randomness).
        """

        # To give more space for float operations
        ENERGY_OFFSET = 1e2

        best_state = self.initial_state
        best_energy = (
            ENERGY_OFFSET
            / playGame([self.initial_state], NeuralNetwork, render=False)[0]
        )

        current_state = best_state
        current_energy = best_energy

        energy_history = [best_energy]

        for _iter in range(self.max_iter):
            # Generate possible states
            states = [
                self.derive_state(current_state) for _ in range(self.states_per_iter)
            ]
            # print(states)
            # Evaluate them
            scores = playGame(
                states,
                NeuralNetwork,
                render=False,
            )

            champion_energy = np.argmax(
                scores
            )  # the best solution is the one with the highest score
            new_energy = ENERGY_OFFSET / scores[champion_energy]

            if (
                new_energy < current_energy
                or self.metropolis(current_energy, new_energy) > random()
            ):
                current_state = states[champion_energy]
                current_energy = new_energy
                energy_history.append(current_energy)
                if new_energy < best_energy:
                    best_state = current_state
                    best_energy = new_energy

            # Cool down
            self.temperature *= 1 - self.cooling_rate

        return best_state, energy_history


if __name__ == "__main__":
    input_weights = np.random.rand(7, 4)
    hidden_weights = np.random.rand(4, 1)
    biases = (random(), random())

    state = (input_weights, hidden_weights, biases)
    # Train
    best_state, _energy_history = SimulatedAnnealing(state).anneal()

    NnWithPrint = lambda state: NeuralNetwork(state, _print=True)

    print(playGame([best_state], NnWithPrint, render=True))

# TODO normalize inputs
