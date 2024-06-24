# pylint: disable=redefined-outer-name

import random
from random import random as random_float

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
    # This is the starting value, it's just a placeholder
    if obType == 2:
        return 0.5
    raise ValueError(f"Unknown obstacle type: {obType}")


def relu(x):
    return np.maximum(0, x)


def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))


def squish_01(x: float) -> float:
    return np.log(x + 1) / np.e


from sklearn.neural_network import MLPClassifier
class ScikitNN(KeyClassifier):
    """For comparison and correctness-checking purposes only"""

    def __init__(self, state: State, activation_threshold=0.55, _print=False):

        self.input_weights = state[0]
        self.hidden_weights = state[1]
        self.input_bias, self.hidden_bias = state[2]
        self.activation_threshold = activation_threshold
        self._print = _print

        self.clf = MLPClassifier(
            hidden_layer_sizes=(4,),
            activation="relu",
            solver="adam",
            alpha=0.0001,
            batch_size="auto",
        )
        weights = [self.input_weights, self.hidden_weights, [1.0]]
        self.clf.coefs_ = weights

        biases = np.array([self.input_bias, self.hidden_bias, 1.0])
        self.clf.intercepts_ = [biases]

        if self._print:
            print(f"{self.clf.coefs_=}")
            print(f"{self.clf.intercepts_=}")

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
            # Normalizing and treating inputs
            obType = normalize_obstable(obType)
            nextObType = normalize_obstable(nextObType)

            # # TODO Experiment with arctan
            # speed = squish_01(speed)
            # distance = squish_01(distance)
            # nextObDistance = squish_01(nextObDistance)
            # obHeight = squish_01(obHeight)
            # nextObHeight = squish_01(nextObHeight)

            prediction = self.clf.predict(
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
                print(f"{prediction=}", end="")

            if prediction >= self.activation_threshold:
                if self._print:
                    print("🔼")
                return "K_UP"
            else:
                if self._print:
                    print("🔽")
                return "K_DOWN"


class NeuralNetwork(KeyClassifier):
    def __init__(self, state: State, activation_threshold=0.55, _print=False):
        # print(state)
        self.input_weights = state[0]
        self.hidden_weights = state[1]
        # self.input_bias, self.hidden_bias = state[2]
        self.input_bias, self.hidden_bias = 0, 0
        self.activation_threshold = activation_threshold
        self._print = _print

    def feedforward(self, inputs: list[float]) -> float:
        """Weights is what will be searched for by the simulated annealing algorithm."""
        hidden_layer = relu(np.dot(inputs, self.input_weights) + self.input_bias)
        output_layer = relu(
            np.dot(hidden_layer, self.hidden_weights) + self.hidden_bias
        )
        return sigmoid(output_layer)[0]

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
        # Normalizing and treating inputs
        obType = normalize_obstable(obType)
        nextObType = normalize_obstable(nextObType)

        # TODO Experiment with arctan
        speed = squish_01(speed)
        distance = squish_01(distance)
        nextObDistance = squish_01(nextObDistance)
        obHeight = squish_01(obHeight)
        nextObHeight = squish_01(nextObHeight)

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
            print(f"{prediction:08.6f}", end="")
        if prediction >= self.activation_threshold:
            if self._print:
                print("🔼")
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
        neural_network=NeuralNetwork,
    ):
        self.initial_state = initial_state
        self.temperature = temperature
        self.initial_temperature = temperature
        self.max_iter = max_iter
        self.cooling_rate = cooling_rate
        self.states_per_iter = states_per_iter
        self.neural_network = neural_network

    def metropolis(self, energy: float, new_energy: float) -> float:
        """
        Probability of accepting a worse solution.
        """

        # TODO Float comparison, but it's fine for now
        if energy == new_energy:
            return 0

        delta = abs(1 / energy - 1 / new_energy)
        metropolis = np.exp(-delta / self.temperature)

        # print(f"{energy=:#.4f} {new_energy=:#.4f} {metropolis=:#}")

        return metropolis

    def derive_state(self, state: State) -> State:
        def perturbation() -> float:
            return random.gauss(mu=0, sigma=self.temperature / self.initial_temperature)

        input_weights = [
            [sigmoid(w + perturbation()) for w in weights] for weights in state[0]
        ]
        hidden_weights = [
            [sigmoid(w + perturbation()) for w in weights] for weights in state[1]
        ]
        biases = tuple(sigmoid(w * perturbation()) for w in state[2])

        new_state = (input_weights, hidden_weights, biases)
        # print(new_state)
        # print(f"On temperature {self.temperature}")
        return new_state

    def score_to_energy(self, score: float) -> float:
        return score

    def anneal(self):
        """
        Anneals the state.

        Returns the best state found.

        The algorithm will run for `max_iter` iterations, each iteration
        will evaluate `self.states_per_iter` possible states at the current temperature.

        A higher energy is better.

        Best overall is not kept to make it more resilient to local minima
        (outliers and randomness).
        """

        current_state = self.initial_state
        current_energy = self.score_to_energy(
            playGame([self.initial_state], self.neural_network, render=False)[0]
        )

        energy_history = [current_energy]

        for epoch in range(self.max_iter):
            # print(f"🌡️ {self.temperature:06.4f} epoch {epoch}")
            # Generate possible states
            states = [
                self.derive_state(current_state) for _ in range(self.states_per_iter)
            ]

            # print(states)

            # Evaluate them
            scores = playGame(
                states,
                self.neural_network,
                # Set to true to see the population playing the game
                render=False,
            )

            # the best solution is the one with the highest energy
            champion_idx = np.argmax(scores)
            new_energy = self.score_to_energy(scores[champion_idx])

            if (
                new_energy > current_energy
                or self.metropolis(current_energy, new_energy) > random_float()
            ):
                print(f"🔥 {new_energy}")
                current_state = states[champion_idx]
                current_energy = new_energy
                energy_history.append(current_energy)

            # Cool down
            # self.temperature *= 1 - self.cooling_rate
            self.temperature = self.initial_temperature * np.exp(
                -self.cooling_rate * epoch
            )
        return current_state, energy_history


if __name__ == "__main__":
    input_weights = np.random.rand(7, 4)
    hidden_weights = np.random.rand(4, 1)
    biases = (random_float(), random_float())

    state = (input_weights, hidden_weights, biases)
    # Train
    # best_state, _energy_history = SimulatedAnnealing(state).anneal()

    # NnWithPrint = lambda state: NeuralNetwork(state, _print=True)

    # print(playGame([best_state], NnWithPrint, render=True))
    best_state, _energy_history = SimulatedAnnealing(
        state, neural_network=ScikitNN
    ).anneal()
    SKNnWithPrint = lambda state: ScikitNN(state, _print=True)
    print(playGame([best_state], SKNnWithPrint, render=True))

    # import seaborn as sns
    # import matplotlib.pyplot as plt

    # sns.lineplot(data=_energy_history)
    # plt.show()
