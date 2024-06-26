# pylint: disable=redefined-outer-name

from typing import Any

from random import random as random_float
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


def normalize_obstable(obType: Obstacle) -> float:
    if isinstance(obType, LargeCactus):
        return 0.0
    if isinstance(obType, SmallCactus):
        return 0.0
    if isinstance(obType, Bird):
        return 1.0
    # This is the starting value, it's just a placeholder
    if obType == 2:
        return 0.5
    raise ValueError(f"Unknown obstacle type: {obType}")


def relu(x):
    return np.maximum(0, x)


def squish_01(x: float) -> float:
    """
    Squish a [0, inf) value between 0 and 1.

    Unlike the sigmoid function, this function is not symmetrical,
    but it does use [0, 0.5) in the image.
    """
    if x < 0:
        print(f"X passed to squish is {x=}")
    return 1 - 1 / (x + 1)


class NeuralNetwork(KeyClassifier):
    """
    The domain for weights is kept [0, 1]
    """

    def __init__(self, state: State, activation_threshold=0.55, _print=False):
        # print(state)
        self.input_weights = state.input_weights
        self.hidden_weights = state.hidden_weights
        self.input_bias, self.hidden_bias = state.biases
        self.activation_threshold = activation_threshold
        self._print = _print

    def feedforward(self, inputs: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        """Weights is what will be searched for by the simulated annealing algorithm."""

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

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
        # Distance can be <0 when the first onscreen obstacle is behind Dino.
        # As I believe this is not relevant for the classifier, I clamp it down.
        distance = relu(distance)

        # print(f"{speed=} {distance=} {nextObDistance=} {obHeight=} {nextObHeight}")
        # speed = squish_01(speed)
        # nextObDistance = squish_01(nextObDistance)
        # obHeight = squish_01(obHeight)
        # nextObHeight = squish_01(nextObHeight)
        # distance = squish_01(distance)

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
        temperature=80,
        max_iter=1000,
        cooling_rate=0.003,
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

        # TODO Float comparison, but it's fine for now since most collision are either
        # at 24 or at 24.5, that's a huge difference
        if energy == new_energy:
            return 0

        delta = abs(energy - new_energy)
        metropolis = np.exp(-delta / self.temperature)

        print(f"{delta=:#.4f} {metropolis=:#}")

        return metropolis

    def derive_state(self, state: State) -> State:
        def perturb(weight: float) -> float:
            def perturbation() -> float:
                # Random float centered on 0 with magnitude <= 2
                p = (random_float() - 0.5) * 2
                # print(f"generated {p}")
                damper = self.temperature / (self.temperature + 1)
                # print(f"{damper=}")
                return p * damper

            p = perturbation()
            # print(f"{p=}")
            result = weight + p
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
                print(f"{current_state=}")
                current_energy = new_energy
                energy_history.append(current_energy)

            # Cool down
            self.boltzmann_cooling(epoch)
        return current_state, energy_history


if __name__ == "__main__":
    # print(playGame([None], TestClassifier, render=True))

    input_weights = np.random.rand(7, 4)
    hidden_weights = np.random.rand(4, 1)
    biases = (random_float(), random_float())

    state = State(input_weights, hidden_weights, biases)
    # Train
    # best_state, energy_history = SimulatedAnnealing(state).anneal()

    params = {
        "temperature": range(50, 250, 20),
        "cooling_rate": range(3, 300, 30),  # / 10000
    }
    params_list = [
        {"temperature": t, "cooling_rate": c / 10000}
        for t in params["temperature"]
        for c in params["cooling_rate"]
    ]

    def run_simulated_annealing(params):
        best_state, energy_history = SimulatedAnnealing(state, **params).anneal()
        # return playGame([best_state], NeuralNetwork, render=False), best_state, params
        return best_state, energy_history[-1], params

    with Pool() as p:
        good_states = p.map(run_simulated_annealing, params_list)
        best_state = max(good_states, key=lambda x: x[1])[0]
        print(good_states)

    # def NnWithPrint(state):
    #     return NeuralNetwork(state, _print=True)

    # print(playGame([best_state], NnWithPrint, render=True))

    # hiscore = max(energy_history)
    # print(f"Hiscore: {hiscore}")

    my_results = manyPlaysResultsTest(30, best_state, NeuralNetwork)

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
