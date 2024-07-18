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
    main,
    manyPlaysResultsComparison,
    manyPlaysResultsTest,
    manyPlaysResultsTrain,
    playGame,
    KeySimplestClassifier,
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

    obheights = []

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
            return 0.0
        if isinstance(obType, SmallCactus):
            return 0.0
        if isinstance(obType, Bird):
            return 1.0
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

        # ObTypes are irrelevant for classification
        # Dino will have to dodge no matter what
        # and can't duck under bird as high as the maximum high cactus
        # (so can't confuse them both, like duck for a high cactus)
        obType = nextObType = 0

        # Distance can be <0 when the first onscreen obstacle is behind Dino.
        # As I believe this is not relevant for the classifier, I clamp it by
        # immediatly switching to the next obstable
        distance = distance if distance > 0 else nextObDistance

        # Dino can duck under 83 bird, but no lower
        obHeight = 100 if obHeight < 83 else 0
        nextObHeight = 100 if nextObHeight < 83 else 0

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
                print(f"{prediction}🔼")
            return "K_UP"
        else:
            if self._print:
                print(f"{prediction}🔽")
            return "K_DOWN"


class SmallNeuralNetwork(NeuralNetwork):
    input_nodes = 3
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
        if distance <= 0:  # if already past, look at next obstacle
            distance = nextObDistance
            obHeight = nextObHeight
        distance -= speed

        # Dino can duck under 83, but no less
        obHeight = 100 if obHeight < 83 else 0

        prediction = self.feedforward(
            np.array(
                [
                    distance,
                    obHeight,
                    speed,
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
    Very useful for debugging
    """

    cactus_height = set()
    bird_height = set()

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
        # print(f"{distance=} {obHeight=}")
        if distance < 300 and distance > 0:
            if isinstance(obType, LargeCactus) or isinstance(obType, SmallCactus):
                if obHeight not in self.cactus_height:
                    print(f"Cactus height: {obHeight}")
                    self.cactus_height.add(obHeight)
                return "K_UP"
            # Bird
            if obHeight not in self.bird_height:
                print(f"Bird height: {obHeight}")
                self.bird_height.add(obHeight)
            if obHeight >= 83:
                return "K_DOWN"
            else:
                return "K_UP"
        else:
            return "K_DOWN"


class SimulatedAnnealing:
    """
    Simulates the cooling of alloys.
    """

    def __init__(
        self,
        initial_state: State,
        temperature=200,
        max_iter=1000,
        cooling_rate=0.003,
        states_per_iter=100,
        neural_network=NeuralNetwork,
        rounds_mean=15,
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

        Best overall is kept and returned but it is preffered to iterate on current_best
        to make it more resilient to local minima (outliers and randomness).
        """
        best_state = current_state = self.initial_state
        best_energy = current_energy = self.score_to_energy(
            manyPlaysResultsTrain(
                self.rounds_mean, [self.initial_state], self.neural_network
            )[0]
        )

        energy_history = [(current_energy, 0)]

        try:
            start_time = time()

            for epoch in range(self.max_iter):
                elapsed_time = time() - start_time
                if elapsed_time >= self.time_limit_secs:
                    break
                print(
                    f"🌡️{self.temperature:06.4f} ⏳{epoch} 🔥{current_energy:0.2f} 🕐 {elapsed_time/50:0.1f}min"
                )
                # Generate possible states
                states = [
                    self.derive_state(current_state)
                    for _ in range(self.states_per_iter)
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
                    energy_history.append((current_energy, epoch))
                    if new_energy > best_energy:
                        best_state = current_state
                        best_energy = current_energy
                        print(f"{current_state=}")

                # Cool down
                self.geometric_cooling(epoch)
        except KeyboardInterrupt:
            print("Annealing halted by user")
        finally:
            return best_state, energy_history


# Useful for tweaking graph generation, and comparisons
saved_energy_history = [
    (23.96132486540519, 0),
    (1116.1726755827444, 1),
    (1395.9142932912216, 2),
    (1375.1835270891968, 3),
    (1586.9006394054522, 4),
    (1440.953647916821, 8),
    (1508.2597547285272, 11),
    (1805.1013635852032, 12),
    (1621.7560118766785, 26),
    (1637.0512363504465, 29),
    (1635.751823776617, 30),
    (1512.091903007007, 32),
    (1751.329891041265, 34),
    (1752.8973039428056, 35),
    (1792.3017285414164, 37),
    (1613.7714769053944, 38),
    (1557.795252886796, 39),
    (1723.8571675270173, 40),
    (1661.512481320981, 41),
    (1895.509318829278, 42),
    (1716.3380563856808, 43),
    (1678.0919498320386, 44),
    (1838.7326728642286, 45),
    (1630.5144330118553, 53),
    (1613.8578850726515, 54),
    (1656.5892731300737, 55),
    (1712.86321860903, 57),
    (1712.7945345735686, 58),
    (1700.917263606663, 59),
    (1816.549676436599, 60),
    (1717.8212132211213, 66),
    (1857.8593111613418, 67),
    (1846.8663271436026, 68),
    (1750.0375292404121, 69),
    (1778.4293711643331, 70),
    (1877.7496669235195, 71),
    (1729.6573068672471, 72),
    (1483.8886379495846, 73),
    (1891.2261684155114, 74),
    (1825.0240689121817, 82),
    (1815.8448783109238, 84),
    (1725.4341127349448, 86),
    (1810.4315837310987, 87),
    (1856.059747217833, 88),
    (1815.0588647926643, 89),
    (1891.7697179665172, 90),
    (1888.4375897361524, 93),
    (1845.7990432044874, 94),
    (1934.4443084397335, 95),
    (1869.4742010217544, 98),
    (1937.8433900753098, 99),
    (1898.5764767186593, 100),
    (1949.8456747727469, 101),
    (1974.5842074165657, 102),
    (2046.229684563926, 103),
    (2122.5458828780093, 107),
    (1960.408969744817, 109),
    (1912.0223541745409, 111),
    (1926.6235513461165, 112),
    (2009.9529915329374, 113),
    (2060.6751852617076, 114),
    (1848.4971323920101, 115),
    (1904.9159862692468, 116),
    (1853.6724986940012, 117),
    (1876.9261033094697, 118),
    (1991.8782706232575, 119),
    (1933.518629500508, 120),
    (1972.6192564612081, 122),
    (2058.75370296663, 123),
    (1961.1989153078703, 124),
    (1970.3159309335908, 125),
    (1988.280698360734, 126),
    (1944.1584761135236, 127),
    (2081.6345043595265, 128),
    (1911.0730480638827, 129),
    (2180.8153233445832, 130),
    (2118.1630147194646, 131),
    (2189.729887369871, 132),
    (2091.4893463833896, 133),
    (2135.6043555214424, 134),
    (2086.2269132660244, 136),
    (2111.4105110666314, 137),
    (2093.2241482229965, 138),
    (2126.0848630567953, 139),
    (2267.1961817133815, 140),
    (2492.3224592341676, 143),
    (2411.189309787909, 147),
    (2351.3492893256634, 148),
    (2359.0136384659704, 151),
    (2334.4700886473156, 152),
    (2683.4787681906846, 153),
    (3174.164117825904, 154),
    (3002.752821303255, 156),
    (3025.254833263981, 157),
    (2963.8322462603555, 158),
    (3144.168476916908, 159),
    (3129.886291527015, 161),
    (3105.7460819268813, 165),
    (2912.0025387880223, 166),
    (3027.8294792177567, 169),
    (3182.576959500438, 170),
    (3162.9781936932045, 171),
    (3140.9228510612247, 172),
    (3154.527342029449, 174),
    (3207.469661041312, 175),
    (3066.2242545208064, 178),
    (2919.2603856746855, 184),
    (3117.856655661615, 186),
    (2949.0535536579914, 211),
    (3097.6080902910307, 213),
    (3178.0922471196895, 221),
    (3086.6996846850598, 224),
    (3142.5717244287825, 226),
    (3149.646486957805, 227),
    (3168.465007899218, 230),
    (3209.971831083747, 232),
    (3194.2902472876603, 233),
    (3192.6426757235163, 234),
    (3171.233615719083, 235),
    (3185.377858928901, 236),
    (3218.9073878212116, 238),
    (3215.7711090222015, 239),
    (3138.660375872519, 240),
    (3138.4244432853725, 241),
    (3152.581136681675, 243),
    (3192.2945490249, 244),
    (3169.9752580619916, 245),
    (3124.3246042082164, 246),
    (3119.4217911498827, 247),
    (3268.3675124282604, 248),
    (3264.397074774502, 249),
    (3177.093002581578, 256),
    (3192.684747455427, 258),
    (3093.0451175387684, 259),
    (3139.240481956536, 260),
    (3206.2987960251585, 261),
    (3178.0037556345987, 264),
    (3208.911685145304, 265),
    (3153.550506994933, 266),
    (3164.800565817419, 267),
    (3120.232403051339, 268),
    (3292.237942204001, 269),
    (3197.953821707267, 270),
    (3132.196485825114, 271),
    (3104.91893789385, 272),
    (3106.468155741552, 273),
    (3097.8581400699404, 275),
    (3062.4873471999454, 276),
    (3126.931239001631, 277),
    (3130.931344906799, 278),
    (3215.1841329424774, 279),
    (3240.3166380887105, 283),
    (3256.4554501867106, 284),
    (3211.4353784468753, 285),
    (3174.045041375848, 286),
    (3187.5934936128974, 287),
    (3139.255872846007, 288),
    (3104.8120536804904, 289),
    (3064.1082024625853, 290),
    (3088.5717577599125, 291),
    (3082.6944104298445, 292),
    (3360.0105149336528, 293),
    (3074.6368323720753, 294),
    (3236.7406555098037, 295),
    (3172.6804682102634, 296),
    (3168.4584111017302, 297),
    (3105.4757219380413, 298),
    (3012.9176626326844, 299),
    (3196.4705688319946, 300),
    (3267.341841179469, 301),
    (3201.4078722936883, 304),
    (3229.437136646804, 306),
    (3097.3421898607853, 307),
    (2754.100186540848, 308),
    (2997.7637636935815, 309),
    (3154.2782344407024, 310),
    (3069.317463259159, 311),
    (3272.3980323247774, 312),
    (3155.358923182638, 315),
    (3212.3109495674294, 317),
    (3213.5954953529126, 318),
    (3185.1475210844915, 320),
    (3404.324974786376, 322),
    (3112.8247830136975, 323),
    (3115.7950761460816, 324),
    (3224.6572679695514, 325),
    (3212.598002482716, 326),
    (3180.4676285665028, 328),
    (3156.190369220285, 329),
    (3231.334606171825, 330),
    (3287.015608971307, 333),
    (3158.4449404274287, 334),
    (3230.753956360714, 336),
    (3023.1653122199314, 337),
    (3094.3945438562428, 338),
    (3226.407561977134, 339),
    (3128.2761030776583, 340),
    (3154.9989982345146, 341),
    (3224.416067487871, 342),
    (3231.178640117327, 343),
    (3260.1895723427338, 344),
    (3180.3918832670506, 350),
    (3194.686068676723, 351),
    (3221.1056583647064, 360),
    (3217.3574178200624, 364),
    (3072.5695755349398, 365),
    (3167.3927670735256, 366),
    (3202.985393643576, 368),
    (3155.361118477446, 369),
    (3253.2849480420896, 370),
    (3188.328548938469, 373),
    (3121.9396194019846, 374),
    (3144.1868903475697, 375),
    (3092.1914644015815, 376),
    (3222.050828316649, 377),
    (3187.2435552484267, 383),
    (3222.2316450651097, 384),
    (3228.2762057437635, 385),
    (3172.8968826657238, 388),
    (3184.452985939532, 389),
    (3209.1033820067128, 390),
    (3254.995427747112, 391),
    (3225.0029291965716, 392),
    (3192.0542235939824, 393),
    (3205.026534610443, 394),
    (3153.3715857042043, 395),
    (3171.288182288412, 396),
    (3264.5676285213626, 397),
    (3334.1806743034, 399),
    (3141.416929012537, 400),
    (3171.211011083687, 401),
    (3039.6842669403195, 402),
    (3162.9080541827757, 403),
    (3375.428783656984, 404),
    (3185.892682464317, 408),
    (3239.2980079965782, 409),
    (3232.8886804513895, 410),
    (3244.2013060240542, 411),
    (3218.0533857935516, 413),
    (3233.36267824005, 416),
    (3235.7993413825507, 417),
    (3328.8447001413333, 418),
    (3229.2338601077968, 428),
    (3172.8907139652542, 432),
    (3142.5301258967056, 434),
    (3373.04884574325, 435),
    (3283.2128574864005, 446),
    (3191.372825046179, 453),
    (3295.427008457001, 454),
    (3306.5500037827205, 455),
    (3162.7831593961846, 457),
    (3280.0328591656134, 458),
    (3181.386306809297, 459),
    (3165.284553722183, 461),
    (3279.3040793510554, 462),
    (3198.2480524553966, 464),
    (3276.7882924454047, 465),
    (3188.386794137197, 475),
    (3181.9238249838418, 476),
    (3238.079328470486, 477),
    (3139.1409608220147, 478),
    (3100.0232632361362, 479),
    (3221.3364996191553, 480),
    (3171.042714452437, 481),
    (3206.838469961476, 482),
    (3206.363030560305, 484),
    (3215.004074268282, 485),
    (3157.3939450599264, 488),
    (3167.8010592694945, 489),
    (3128.85498084168, 491),
    (3267.559807550763, 492),
    (3340.0505769936717, 493),
    (3199.947504593712, 494),
    (3165.8037923255474, 496),
    (3148.3085268661025, 497),
    (3227.239828048235, 498),
    (3165.096098456541, 499),
    (3227.8278717358644, 500),
    (3258.080485699854, 501),
    (3318.0981268359023, 504),
    (3285.324913883367, 508),
    (3231.6673344547444, 516),
    (3118.7047084707997, 517),
    (3135.211164316672, 518),
    (3136.175428264415, 519),
    (3125.973117048443, 520),
    (3177.1695504173426, 521),
    (3258.1714332442034, 522),
    (3288.8736600912207, 523),
    (3286.0526612471613, 526),
    (3287.10072433992, 528),
    (3229.352630599031, 536),
    (3066.8778561286445, 539),
    (3134.5314922090206, 540),
    (3235.2575563329183, 541),
    (3180.918052880267, 543),
    (3349.2496097014146, 544),
    (3207.1823956041, 557),
    (3309.4773810646893, 558),
    (3103.765492250969, 564),
    (3186.270216040991, 565),
    (3199.690359267677, 566),
    (3186.973667671333, 567),
    (3164.2178531450936, 568),
    (3152.5065749981327, 569),
    (3163.641593314256, 570),
    (3275.309642471872, 571),
    (3159.0969175695295, 573),
    (3195.0182116263845, 574),
    (3225.1675158246226, 575),
    (3197.8216148843835, 576),
    (3222.4325801813097, 577),
    (3175.5442388504175, 578),
    (3273.87276825278, 579),
    (3128.6044474819773, 582),
    (3145.507329785267, 583),
]

state_2000 = State(
    input_weights=[
        [-650.49642471, -345.74577463],
        [705.21388508, 889.78615749],
        [2475.14087666, 1043.01354322],
    ],
    hidden_weights=[[-434.79761561], [322.39540003]],
)
state_3000 = State(
    input_weights=[
        [-318.96541139, -166.16543242],
        [243.60032247, 322.07072038],
        [2606.21566746, 1314.45255703],
    ],
    hidden_weights=[[-678.27825774], [757.5212445]],
)

# This was achieved with a long (~8hr) training session
# Useful for tweaking graph generation
# not from the same training session as 2000 and 3000 above,
# which are from the same training session within themselves
known_good_state = State(
    input_weights=[
        [-503.19814273, -356.11155497],
        [947.65566578, 71.59460554],
        [3122.44492769, 2571.82767144],
    ],
    hidden_weights=[[931.31137591], [-4191.77844538]],
)


if __name__ == "__main__":
    # To make results comparable
    # seed(0)
    # np.random.seed(0)

    ## Full NeuralNetwork (7-4-1)
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

    ## Small NeuralNetwork (3-2-1)

    best_state, energy_history = SimulatedAnnealing(
        initial_state=State.create(SmallNeuralNetwork),
        neural_network=SmallNeuralNetwork,
        time_limit_secs=60 * 60 * 12,
        max_iter=1000,  # will likely reach 1000 way before 12hr
    ).anneal()

    # energy_history = saved_energy_history

    energy_history_expanded = []
    for (energy, epoch), (_, nextepoch) in zip(energy_history, energy_history[1:]):
        energy_history_expanded.extend([energy] * (nextepoch - epoch))

    # print(energy_history_expanded)

    # manyPlaysResultsComparison(
    #     1,
    #     [(state_2000, SmallNeuralNetwork), (state_3000, SmallNeuralNetwork)],
    #     render=True,
    # )

    # print(energy_history_expanded)

    sns.set_theme(style="darkgrid")

    sns.lineplot(x=range(len(energy_history_expanded)), y=energy_history_expanded)

    # Only for saved energy history
    # hightlight_epochs = [152, 153]
    # hightlight_values = [energy_history_expanded[i] for i in hightlight_epochs]
    # sns.scatterplot(
    #     x=hightlight_epochs,
    #     y=hightlight_values,
    #     label="Highlighted epochs",
    #     color="red",
    # )
    plt.title("Energy history")
    plt.xlabel("Epoch")
    plt.ylabel("Energy")
    plt.show()

    # print(best_state)

    # # test
    # print(playGame([best_state], SmallNeuralNetwork, render=True))

    ## Test classifier

    # best_state = None

    # print(playGame([best_state], TestClassifier, render=False))

    ###

    setup = [
        (best_state, SmallNeuralNetwork),
        ([(15, 250), (19, 386), (20, 450), (1000, 550)], KeySimplestClassifier),
    ]

    play_rounds = 30

    my_results, simplest_results = manyPlaysResultsComparison(play_rounds, setup)
    prof_results = [
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

    sns.boxplot(
        data={
            "author": my_results[0],
            "professor": prof_results,
            "simplest": simplest_results[0],
        }
    )
    plt.show()
