from ioh import get_problem, ProblemClass
from ioh import logger
import random
import numpy as np
from ast import literal_eval
import time

# Parameter settings.
PROBLEM_ID = 18
RUNS = 20
DIMENSION = 10
POPULATION_SIZE = 10
MUTATION_RATE = 0.1
UPDATE_RATIO = 0.1
SEED = 42

ga_args_dict = {"population_size": POPULATION_SIZE, "mutation_rate": MUTATION_RATE,
                "update_ratio": UPDATE_RATIO, "dim": DIMENSION}

class GA:
    """The generic algorithm implementation."""

    def __init__(self, population_size, mutation_rate, update_ratio, dim, seed=42, budget=5000,
                 tournament_k=5, tournament_p=0.5):
        """
        :param population_size: int. Only consider the GA whose population size is greater than 1.
        :param mutation_rate: float.
        :param update_ratio: float. The ratio of updating the population.
        :param dim: int. Dimension of the search space.
        :param seed: int. Random seed. (default:42)
        :param budget: int. Budget for each run. (default: 5000)
        :param tournament_k: int. The number of individuals used in tournament selection. (default: 5)
        :param tournament_p: float. The probability of selecting a individual in tournament selection. (default: 0.5)
        """
        assert population_size > 1, "The population size should be an integer greater than 1."
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.update_number = max(int(update_ratio * population_size), 1)
        self.dim = dim
        self.seed = seed
        self.budget = budget
        self.fix_budget = budget
        self.tournament_k = tournament_k
        self.tournament_p = tournament_p
        self.selection_probabilities = None
        self.selection_table = {
            "prop": self._proportional_selection,
            "rank": self._rank_based_selection,
            "tour": self._tournament_selection,
        }

    class Instance:
        """An individual."""

        flip = {"0": "1", "1": "0"}

        def __init__(self, outer_instance, string=None):
            """
            :param outer_instance: GA. An instance of GA.
            :param string: str. The value of the instance. (default: None)
            """
            self.outer_instance = outer_instance
            if string is None:
                self.value = self._instance_random_generator()
            else:
                self.value = string
            self.fitness = self.outer_instance.func(self.numerical_value)
            self.outer_instance.budget -= 1

        def _instance_random_generator(self):
            """Generate an instance.
            :return str.
            """
            random.seed(self.outer_instance.seed)
            self.outer_instance.seed += 1
            return "".join(random.choices(["0", "1"], k=self.outer_instance.dim))

        @property
        def numerical_value(self):
            """str -> np.array(int)"""
            return np.array(list(map(literal_eval, list(self.value))))

        def mutation(self, p):
            """
            :param p: float. The mutation probability.
            """
            value = list(self.value)
            for i in range(self.outer_instance.dim):
                random.seed(self.outer_instance.seed)
                self.outer_instance.seed += 1
                if random.random() < p:
                    value[i] = GA.Instance.flip[value[i]]
            self.value = "".join(value)
            self.fitness = self.outer_instance.func(self.numerical_value)
            self.outer_instance.budget -= 1

    def _initialization(self):
        """Generate the initial population."""
        self.population = list()
        for _ in range(self.population_size):
            self.population.append(GA.Instance(self))
        self.population.sort(key=lambda x: x.fitness)

    def _selection(self, strategy):
        """Selection operator.
        :return: np.array.
            An array of two selected parents.
        """
        if strategy == "prop" or strategy == "rank":
            if self.selection_probabilities is None:
                self.selection_table[strategy]()
            np.random.seed(self.seed)
            self.seed += 1
            return np.random.choice(np.array(self.population), size=2, replace=False, p=self.selection_probabilities)
        elif strategy == "tour":
            return self.selection_table[strategy]()
        else:
            raise KeyError(f"{strategy} is an invalid selection strategy.")

    def _proportional_selection(self):
        """The selected probability is proportional to the fitness."""
        probability_arr = list()
        for item in self.population:
            probability_arr.append(item.fitness)
        probability_arr = np.array(probability_arr, dtype=np.float64)
        self.selection_probabilities = probability_arr / probability_arr.sum()

    def _rank_based_selection(self):
        """The selected probability is proportional to the ranking of the fitness."""

    def _tournament_selection(self):
        """Tournament selection implementation.
        :return: np.array.
            An array of two selected parents.
        """

    def _one_point_crossover(self, parents_arr):
        """Perform the crossover on one random point.
        :param parents_arr:
        :return: GA.Instance.
        """
        assert parents_arr.shape[0] == 2, "The shape of parameter parents_arr is invalid."
        random.seed(self.seed)
        self.seed += 1
        loc = random.randint(1, self.dim - 1)
        return GA.Instance(self, parents_arr[0].value[:loc] + parents_arr[1].value[loc:])

    def _mutation(self, instance):
        """Mutation operator."""
        instance.mutation(self.mutation_rate)

    def _ga_subroutine(self):
        """Update the population partially."""
        update_number = self.update_number
        while self.budget > 0 and update_number > 0:
            parents = self._selection("prop")
            child = self._one_point_crossover(parents)
            self.population.pop(0)
            self.population.append(child)
            self.population.sort(key=lambda x: x.fitness)
            update_number -= 1
        for instance in self.population:
            if self.budget > 0:
                self._mutation(instance)
        self.population.sort(key=lambda x: x.fitness)
        self.selection_probabilities = None

    def __call__(self, func):
        """Implement the genetic algorithm."""
        self.func = func
        self._initialization()

        while self.budget > 0:
            self._ga_subroutine()

        func.reset()
        opt = self.population[-1]
        return opt.fitness, opt.value

    def reset(self):
        """Reset the state."""
        self.budget = self.fix_budget


def ga_main(ga_args, problem_id):
    start = time.time()
    algorithm = GA(**ga_args)
    problem = get_problem(fid=problem_id, dimension=ga_args["dim"], problem_class=ProblemClass.PBO)
    ga_logger = logger.Analyzer(root="../data",
                                folder_name="run",
                                algorithm_name=f"GA: F{PROBLEM_ID}",
                                algorithm_info=f"GA: F{PROBLEM_ID}"
                                )
    problem.attach_logger(ga_logger)

    for run in range(RUNS):
        print(f"Runs: {run + 1}")
        algorithm(problem)
        problem.reset()
        algorithm.reset()

    end = time.time()
    print("The program takes %s seconds" % (end - start))


ga_main(ga_args_dict, 18)
