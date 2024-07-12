# pylint: disable=redefined-outer-name
from __future__ import annotations

from dataclasses import dataclass
from random import random as random_float
from typing import Any, Type, Self
from time import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
    # biases: Biases

    @classmethod
    def create(cls, nn: Type[NeuralNetwork]) -> Self:
        input_weights = np.random.rand(nn.input_nodes, nn.hidden_nodes)
        hidden_weights = np.random.rand(nn.hidden_nodes, nn.output_nodes)
        # biases = (random_float(), random_float())
        return cls(input_weights, hidden_weights)


class NeuralNetwork(KeyClassifier):
    input_nodes = 7
    hidden_nodes = 4
    output_nodes = 1

    def __init__(self, state: State, activation_threshold=0.55, _print=False):
        self.activation_threshold = activation_threshold
        self._print = _print

        self.input_weights = state.input_weights
        self.hidden_weights = state.hidden_weights
        # self.input_bias, self.hidden_bias = state.biases

    @classmethod
    def initiate_with_state(cls) -> Self:
        return cls(State.create(cls))

    @classmethod
    def normalize_obstable(cls, obType: Obstacle) -> float:
        if isinstance(obType, LargeCactus):
            return 1.0
        if isinstance(obType, SmallCactus):
            return 1.0
        if isinstance(obType, Bird):
            return 0.0
        # This is the starting value, it's just a placeholder
        # Which is == 2:
        return 0.5

    @classmethod
    def relu(cls, x):
        return np.maximum(0, x)

    @classmethod
    def map_0inf_to_01(cls, x: int):
        return 1 - np.e ** (-x / 100)

    def feedforward(self, inputs: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        # hidden_layer = self.relu(np.dot(inputs, self.input_weights) + self.input_bias)
        # output_layer = self.relu(
        #     np.dot(hidden_layer, self.hidden_weights) + self.hidden_bias
        # )
        hidden_layer = self.relu(np.dot(inputs, self.input_weights))
        output_layer = self.relu(np.dot(hidden_layer, self.hidden_weights))
        return sigmoid(output_layer[0])

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
        # speed = self.map_0inf_to_01(speed)
        # distance = self.map_0inf_to_01(distance)
        # obHeight = self.map_0inf_to_01(obHeight)
        # nextObDistance = self.map_0inf_to_01(nextObDistance)
        # nextObHeight = self.map_0inf_to_01(nextObHeight)

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
        # speed = self.map_0inf_to_01(speed)
        # distance = self.map_0inf_to_01(distance)
        # obHeight = self.map_0inf_to_01(obHeight)
        # nextObDistance = self.map_0inf_to_01(nextObDistance)
        # nextObHeight = self.map_0inf_to_01(nextObHeight)

        # distance = 200 / distance
        # if obType == 0:
        #     print(f"{obType=}, {obHeight=}")
        # obHeight = 123 - obHeight
        obHeight = 100 if obHeight < 100 else 0
        # obHeight = obHeight if obHeight < 100 else 0

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


class TinyNeuralNetwork(NeuralNetwork):
    input_nodes = 3
    hidden_nodes = output_nodes = 1

    def feedforward(self, inputs: np.ndarray[Any, np.dtype[np.float64]]) -> float:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        output_layer = self.relu(np.dot(inputs, self.input_weights))
        return sigmoid(output_layer[0])

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
        distance = distance if distance > 0 else 500  # Some arbitraty high distance

        # print(f"{speed=} {distance=} {nextObDistance=} {obHeight=} {nextObHeight}")
        # speed = squish_01(speed)
        # speed = self.map_0inf_to_01(speed)
        # distance = self.map_0inf_to_01(distance)
        # obHeight = self.map_0inf_to_01(obHeight)
        # nextObDistance = self.map_0inf_to_01(nextObDistance)
        # nextObHeight = self.map_0inf_to_01(nextObHeight)

        # distance = 200 / distance
        # if obType == 0:
        #     print(f"{obType=}, {obHeight=}")
        obHeight = 123 - obHeight
        # obHeight = 100 if obHeight < 100 else 0
        # obHeight = obHeight if obHeight < 100 else 0

        prediction = self.feedforward(
            np.array(
                [
                    distance,
                    obHeight,
                    # speed,
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
        temperature=200,
        max_iter=2000,
        cooling_rate=0.003,
        states_per_iter=100,
        neural_network=NeuralNetwork,
        rounds_mean=5,
        time_limit_secs=300,
    ):
        self.initial_state = initial_state
        self.temperature = temperature
        self.initial_temperature = temperature
        self.max_iter = max_iter
        self.cooling_rate = cooling_rate
        self.states_per_iter = states_per_iter
        self.neural_network = neural_network
        self.rounds_mean = rounds_mean
        self.time_limit_secs = time_limit_secs

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
                damper = self.temperature / (self.temperature + 10)
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
        # biases = (perturb(state.biases[0]), perturb(state.biases[1]))

        new_state = State(input_weights, hidden_weights)
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

        start_time = time()

        for epoch in range(self.max_iter):
            elapsed_time = time() - start_time
            if elapsed_time >= self.time_limit_secs:
                break
            print(
                f"🌡️{self.temperature:06.4f} ⏳{epoch} 🔥{current_energy:0.2f} 🕐 {elapsed_time:0.1f}"
            )
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

            scores_std = np.std(scores)

            print(f"{scores_std=}")

            # the best solution is the one with the highest energy
            champion_idx = np.argmax(scores)
            new_energy = self.score_to_energy(scores[champion_idx])

            if (
                new_energy > current_energy
                or self.metropolis(current_energy, new_energy) > random_float()
            ):
                current_state = states[champion_idx]
                current_energy = new_energy
                energy_history.append(current_energy)
                if new_energy > best_energy:
                    best_state = current_state
                    best_energy = current_energy
                    print(f"{current_state=}")

            # Cool down
            self.boltzmann_cooling(epoch)
        return best_state, energy_history


if __name__ == "__main__":
    # To make results comparable
    # seed(0)
    # np.random.seed(0)

    # # Train
    # best_state, energy_history = SimulatedAnnealing(
    #     initial_state=State.create(NeuralNetwork),
    #     neural_network=NeuralNetwork,
    #     time_limit_secs=600,
    # ).anneal()

    # def NnWithPrint(state):
    #     return NeuralNetwork(state, _print=True)

    # print(best_state)

    # # test
    # print(playGame([best_state], NnWithPrint, render=True))

    # hiscore = max(energy_history)
    # print(f"Hiscore: {hiscore}")

    # my_results, my_score = manyPlaysResultsTest(30, best_state, NeuralNetwork)
    # print(f"{my_score=}")

    ###

    best_state, energy_history = SimulatedAnnealing(
        initial_state=State.create(SmallNeuralNetwork),
        neural_network=SmallNeuralNetwork,
        time_limit_secs=60 * 15,
    ).anneal()

    def NnWithPrint(state):
        return SmallNeuralNetwork(state, _print=True)

    print(best_state)

    # test
    print(playGame([best_state], NnWithPrint, render=True))

    ###

    # best_state, energy_history = SimulatedAnnealing(
    #     initial_state=State.create(TinyNeuralNetwork),
    #     neural_network=TinyNeuralNetwork,
    #     time_limit_secs=300,
    # ).anneal()

    # def NnWithPrint(state):
    #     return TinyNeuralNetwork(state, _print=True)

    # print(best_state)

    # # test
    # print(playGame([best_state], NnWithPrint, render=True))

    hiscore = max(energy_history)
    print(f"Hiscore: {hiscore}")

    my_results, my_score = manyPlaysResultsTest(30, best_state, SmallNeuralNetwork)
    print(f"{my_score=}")

    mean_std = np.mean(my_results) - np.std(my_results)
    print(f"final score = {mean_std}")

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

    # Os 30 resultados de cada agente devem ser apresentados textualmente em uma tabela, junto com a média e desvio
    # padrão. Os (p-values) do teste t pareado com amostras independentes (scipy.stats.ttest_ind) e
    # do teste não paramétrico de wilcoxon devem ser apresentados também indicando se existe
    # diferença significativa entre os métodos em um nível de 95% de significância.

    sns.boxplot(data={"mine": my_results, "prof": prof_result})
    plt.show()
