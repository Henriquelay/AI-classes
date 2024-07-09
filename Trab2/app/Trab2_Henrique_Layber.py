# pylint: disable=redefined-outer-name

from typing import Any

from random import random as random_float, seed
from dataclasses import dataclass

from multiprocessing import Pool


import numpy as np

from Dino.dinoAIParallel import (
    Bird,
    KeyClassifier,
    LargeCactus,
    Obstacle,
    SmallCactus,
    manyPlaysResultsTest,
    manyPlaysResultsTrain,
    playGame,
)

type InputWeights = np.ndarray[
    Any, np.dtype[np.float64]
]  # 7x4 matrix of input x hidden weights
type HiddenWeights = np.ndarray[
    Any, np.dtype[np.float64]
]  # 4x1 matrix of hidden x output weights
type Biases = tuple[float, float]  # layer biases


@dataclass
class State:
    input_weights: InputWeights
    hidden_weights: HiddenWeights
    biases: Biases


class NeuralNetwork(KeyClassifier):
    def __init__(self, state: State, activation_threshold=0.55, _print=False):
        # print(state)
        self.input_weights = state.input_weights
        self.hidden_weights = state.hidden_weights
        self.input_bias, self.hidden_bias = state.biases
        self.activation_threshold = activation_threshold
        self._print = _print

    @classmethod
    def normalize_obstable(cls, obType: Obstacle) -> float:
        if isinstance(obType, LargeCactus):
            return 1.0
        if isinstance(obType, SmallCactus):
            return 1.0
        if isinstance(obType, Bird):
            return 0.0
        # This is the starting value, it's just a placeholder
        if obType == 2:
            return 0.5
        raise ValueError(f"Unknown obstacle type: {obType}")

    @classmethod
    def relu(cls, x):
        return np.maximum(0, x)

    @classmethod
    def map_0inf_to_01(cls, x: int):
        return 1 - np.e ** (-x / 100)

    def feedforward(self, inputs: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        hidden_layer = self.relu(np.dot(inputs, self.input_weights) + self.input_bias)
        output_layer = self.relu(
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
        obType = self.normalize_obstable(obType)
        nextObType = self.normalize_obstable(nextObType)
        # Distance can be <0 when the first onscreen obstacle is behind Dino.
        # As I believe this is not relevant for the classifier, I clamp it down.
        distance = self.relu(distance)

        # print(f"{speed=} {distance=} {nextObDistance=} {obHeight=} {nextObHeight}")
        # speed = squish_01(speed)
        speed = self.map_0inf_to_01(speed)
        distance = self.map_0inf_to_01(distance)
        obHeight = self.map_0inf_to_01(obHeight)
        nextObDistance = self.map_0inf_to_01(nextObDistance)
        nextObHeight = self.map_0inf_to_01(nextObHeight)

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
                dtype=np.float64,
            )
        )

        if prediction >= self.activation_threshold:
            if self._print:
                print(f"{prediction:08.6f}🔼")
            return "K_UP"
        else:
            if self._print:
                print(f"{prediction:08.6f}🔽")
            return "K_DOWN"


class SmallNeuralNetwork(NeuralNetwork):
    input_nodes = 4
    hidden_nodes = 2
    output_nodes = 1

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
        obType = self.normalize_obstable(obType)
        # nextObType = self.normalize_obstable(nextObType)
        # Distance can be <0 when the first onscreen obstacle is behind Dino.
        # As I believe this is not relevant for the classifier, I clamp it down.
        distance = self.relu(distance)

        # print(f"{speed=} {distance=} {nextObDistance=} {obHeight=} {nextObHeight}")
        # speed = squish_01(speed)
        speed = self.map_0inf_to_01(speed)
        distance = self.map_0inf_to_01(distance)
        obHeight = self.map_0inf_to_01(obHeight)
        # nextObDistance = self.map_0inf_to_01(nextObDistance)
        # nextObHeight = self.map_0inf_to_01(nextObHeight)

        prediction = self.feedforward(
            np.array(
                [
                    distance,
                    obHeight,
                    speed,
                    obType,
                ],
                dtype=np.float64,
            )
        )

        if prediction >= self.activation_threshold:
            if self._print:
                print(f"{prediction:08.6f}🔼")
            return "K_UP"
        else:
            if self._print:
                print(f"{prediction:08.6f}🔽")
            return "K_DOWN"


class TestClassifier(KeyClassifier):
    """
    The domain for weights is kept [0, 1]
    """

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
        print(f"{distance=} {obHeight=}")
        if distance < 300 and distance > 0:
            if isinstance(obType, LargeCactus) or isinstance(obType, SmallCactus):
                return "K_UP"
            # Bird
            if obHeight > 50:
                return "K_DOWN"
            else:
                return "K_UP"
        else:
            return "K_DOWN"


class SimulatedAnnealing:
    """Simulated Annealing algorithm."""

    def __init__(
        self,
        initial_state: State,
        temperature=400,
        max_iter=10000,
        cooling_rate=0.005,
        states_per_iter=100,
        neural_network=NeuralNetwork,
        rounds_mean=10,
    ):
        self.initial_state = initial_state
        self.temperature = temperature
        self.initial_temperature = temperature
        self.max_iter = max_iter
        self.cooling_rate = cooling_rate
        self.states_per_iter = states_per_iter
        self.neural_network = neural_network
        self.rounds_mean = rounds_mean

    def metropolis(self, energy: float, new_energy: float) -> float:
        """
        Probability of accepting a worse solution.
        """
        delta = abs(energy - new_energy)
        metropolis = np.exp(-delta / self.temperature)

        print(f"{delta=:#.4f} {metropolis=:#}")

        return metropolis

    def derive_state(self, state: State) -> State:
        def perturb(weight: float) -> float:
            def perturbation() -> float:
                damper = self.temperature / (self.temperature + 1)
                return np.random.uniform(-damper, damper)

            p = perturbation()
            # print(f"{p=}")
            result = weight + self.temperature * p
            return result

        input_weights = np.array(
            [[perturb(w) for w in weights] for weights in state.input_weights]
        )
        hidden_weights = np.array(
            [[perturb(w) for w in weights] for weights in state.hidden_weights]
        )
        biases = (perturb(state.biases[0]), perturb(state.biases[1]))

        new_state = State(input_weights, hidden_weights, biases)
        # print(new_state)
        # print(f"On temperature {self.temperature}")
        return new_state

    def score_to_energy(self, score: float) -> float:
        return score

    def boltzmann_cooling(self, epoch: int):
        self.temperature = self.initial_temperature / np.log(epoch + 2)

    def exponential_cooling(self, _):
        self.temperature = self.temperature * (1 - self.cooling_rate)

    def geometric_cooling(self, epoch: int):
        self.temperature = self.initial_temperature / (1 + epoch * self.cooling_rate)

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

        best_state = current_state = self.initial_state
        best_energy = current_energy = self.score_to_energy(
            manyPlaysResultsTrain(
                self.rounds_mean, [self.initial_state], self.neural_network
            )[0]
        )

        energy_history = [current_energy]

        for epoch in range(self.max_iter):
            print(f"🌡️{self.temperature:06.4f} ⏳{epoch} 🔥{current_energy}")
            # Generate possible states
            states = [
                self.derive_state(current_state) for _ in range(self.states_per_iter)
            ]

            # Evaluate them
            scores = manyPlaysResultsTrain(
                self.rounds_mean,
                states,
                self.neural_network,
                # Set to true to see the population playing the game
            )

            # the best solution is the one with the highest energy
            champion_idx = np.argmax(scores)
            new_energy = self.score_to_energy(scores[champion_idx])

            if (
                new_energy > current_energy
                or self.metropolis(current_energy, new_energy) > random_float()
            ):
                current_state = states[champion_idx]
                # print(f"{current_state=}")
                current_energy = new_energy
                energy_history.append(current_energy)
                if new_energy > best_energy:
                    best_state = current_state
                    best_energy = current_energy

            # Cool down
            self.boltzmann_cooling(epoch)
        return best_state, energy_history


if __name__ == "__main__":
    # To make results comparable
    # seed(0)
    # np.random.seed(0)

    # input_weights = np.random.rand(7, 4)
    # hidden_weights = np.random.rand(4, 1)
    # biases = (random_float(), random_float())

    # state = State(input_weights, hidden_weights, biases)
    # # Train
    # best_state, energy_history = SimulatedAnnealing(state).anneal()

    # def NnWithPrint(state):
    #     return NeuralNetwork(state, _print=True)

    # print(playGame([best_state], NnWithPrint, render=True))

    input_weights = np.random.rand(
        SmallNeuralNetwork.input_nodes, SmallNeuralNetwork.hidden_nodes
    )
    hidden_weights = np.random.rand(
        SmallNeuralNetwork.hidden_nodes, SmallNeuralNetwork.output_nodes
    )
    biases = (random_float(), random_float())

    state = State(input_weights, hidden_weights, biases)
    # Train
    best_state, energy_history = SimulatedAnnealing(
        state, neural_network=SmallNeuralNetwork, max_iter=1000
    ).anneal()

    def SmallNnWithPrint(state):
        return SmallNeuralNetwork(state, _print=True)

    print(playGame([best_state], SmallNnWithPrint, render=True))

    hiscore = max(energy_history)
    print(f"Hiscore: {hiscore}")

    my_results, my_score = manyPlaysResultsTest(30, best_state, SmallNeuralNetwork)
    print(f"{my_score=}")

    prof_result = [
        1214.0,
        759.5,
        1164.25,
        977.25,
        1201.0,
        930.0,
        1427.75,
        799.5,
        1006.25,
        783.5,
        728.5,
        419.25,
        1389.5,
        730.0,
        1306.25,
        675.5,
        1359.5,
        1000.25,
        1284.5,
        1350.0,
        751.0,
        1418.75,
        1276.5,
        1645.75,
        860.0,
        745.5,
        1426.25,
        783.5,
        1149.75,
        1482.25,
    ]

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.boxplot(data={"mine": my_results, "prof": prof_result})
    plt.show()
