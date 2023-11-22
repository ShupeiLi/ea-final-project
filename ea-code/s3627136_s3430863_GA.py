from ioh import get_problem, logger, ProblemClass
import random
import numpy as np
from ast import literal_eval
import time
import copy

# Parameter settings.
RUNS = 20
DIMENSION = 50
SEED = 42
BUDGET = 5000

# Hyperparameters
F18_POPULATION_SIZE = 20
F18_MUTATION_RATE = -1.0
F18_UPDATE_RATIO = 0.4

F19_POPULATION_SIZE = 20
F19_MUTATION_RATE = 0.02
F19_UPDATE_RATIO = 0.4

ga_f18_args_dict = {"population_size": F18_POPULATION_SIZE, "mutation_rate": F18_MUTATION_RATE,
                    "update_ratio": F18_UPDATE_RATIO, "dim": DIMENSION, "budget": BUDGET,
                    "seed": SEED}
ga_f19_args_dict = {"population_size": F19_POPULATION_SIZE, "mutation_rate": F19_MUTATION_RATE,
                    "update_ratio": F19_UPDATE_RATIO, "dim": DIMENSION, "budget": BUDGET,
                    "seed": SEED}


class GA:
    """The generic algorithm implementation."""

    def __init__(self, population_size, update_ratio, dim, problem, seed, budget, mutation_rate=-1.0,
                 tournament_p=0.8):
        """
        :param population_size: int. Only consider the GA whose population size is greater than 1.
        :param update_ratio: float. The ratio of updating the population.
        :param dim: int. Dimension of the search space.
        :param problem: ioh Problem instance.
        :param seed: int. Random seed.
        :param budget: int. Budget for each run.
        :param mutation_rate: float. If using the fixed mutation rate, it should be a value in (0, 1). If
                              the mutation rate equals -1.0, use the annealing schedule. (default: -1.0)
        :param tournament_p: float. The probability of selecting a individual in tournament selection. (default: 0.8)
        """
        assert population_size > 1, "The population size should be an integer greater than 1."
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.update_number = max(int(update_ratio * population_size), 1)
        self.dim = dim
        self.seed = seed
        self.fix_seed = seed
        self.problem = problem
        self.budget = budget
        self.fix_budget = budget
        self.tournament_k = population_size
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
        if strategy in self.selection_table.keys():
            if self.selection_probabilities is None:
                self.selection_table[strategy]()
            np.random.seed(self.seed)
            self.seed += 1
            return np.random.choice(np.array(self.population), size=2, replace=False, p=self.selection_probabilities)
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
        rank_arr = np.arange(1, self.population_size + 1)
        self.selection_probabilities = (2 * rank_arr) / (self.population_size * (self.population_size + 1))

    def _tournament_selection(self):
        """Tournament selection implementation."""
        np.random.seed(self.seed)
        self.seed += 1
        index = np.random.choice(np.arange(self.population_size), size=self.tournament_k, replace=False)
        probability = list()
        pointer = 0
        for i in range(self.population_size - 1, -1, -1):
            if i in index:
                probability.append(self.tournament_p * ((1 - self.tournament_p) ** pointer))
                pointer += 1
            else:
                probability.append(0)
        probability.reverse()
        probability = np.array(probability)
        self.selection_probabilities = probability / probability.sum()

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
        if self.mutation_rate < 0:
            if self.budget < 3000:
                instance.mutation(1 / self.dim)
            else:
                instance.mutation(0.1)
        else:
            instance.mutation(self.mutation_rate)

    def _ga_subroutine(self):
        """Update the population partially."""
        update_number = self.update_number
        while self.budget > 0 and update_number > 0:
            # prop -> rank -> tour
            if (self.fix_budget - self.budget) < 2000:
                parents = self._selection("prop")
            elif 2000 <= (self.fix_budget - self.budget) < 3000:
                parents = self._selection("rank")
            else:
                parents = self._selection("tour")
            self.selection_probabilities = None
            child = self._one_point_crossover(parents)
            if self.population[0].fitness < child.fitness:
                self.population.pop(0)
                self.population.append(child)
                self.population.sort(key=lambda x: x.fitness)
                update_number -= 1
        for instance in self.population:
            if self.budget > 0:
                self._mutation(instance)
        self.population.sort(key=lambda x: x.fitness)

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
        self.fix_seed += 1
        self.seed = self.fix_seed


def ga_main(ga_args, problem_id, info=None):
    """Run the GA algorithm.
    :param ga_args: dict. Parameters defined in GA algorithm.
    :param problem_id: int. 18 or 19.
    :param info: dict. The information of the algorithm.
    """
    start = time.time()
    problem = get_problem(fid=problem_id, dimension=ga_args["dim"], instance=1, problem_class=ProblemClass.PBO)
    ga_args["problem"] = problem
    algorithm = GA(**ga_args)
    if info is not None:
        ga_logger = logger.Analyzer(root="data",
                                    folder_name=f"{info['run']} {info['param']} run",
                                    algorithm_name=f"s3627136_s3430863_GA: F{problem_id}, {info['param']}",
                                    algorithm_info=f"Practical assignment of the EA course."
                                    )
    else:
        ga_logger = logger.Analyzer(root="data",
                                    folder_name="run",
                                    algorithm_name=f"s3627136_s3430863_GA: F{problem_id}",
                                    algorithm_info=f"Practical assignment of the EA course."
                                    )
    problem.attach_logger(ga_logger)

    if info is not None:
        print(f"INFO: {info['param']}")
    for run in range(RUNS):
        print(f"Runs: {run + 1}")
        algorithm(problem)
        problem.reset()
        algorithm.reset()

    ga_logger.close()
    end = time.time()
    print("The program takes %s seconds" % (end - start))


def reproduce_report(args_dict, problem_id, param):
    """Reproduce fine-tuning results in our report."""
    if param == "p_size":
        pop_dict = copy.deepcopy(args_dict)
        for pop_k in [2, 5, 10, 20, 25]:
            pop_dict["population_size"] = pop_k
            ga_main(pop_dict, problem_id, {"run": "population size", "param": f"p_size={pop_k}"})
    elif param == "m_rate":
        mu_dict = copy.deepcopy(args_dict)
        mu_char = [0.02, 0.05, 0.1, "piecewise"]
        mu_lst = [1 / DIMENSION, 0.05, 0.1, -1.0]
        for i in range(len(mu_lst)):
            mu_dict["mutation_rate"] = mu_lst[i]
            ga_main(mu_dict, problem_id, {"run": "mutation rate", "param": f"m_rate={mu_char[i]}"})
    elif param == "u_rate":
        u_dict = copy.deepcopy(args_dict)
        for ur in [0.2, 0.4, 0.8, 1]:
            u_dict["update_ratio"] = ur
            ga_main(u_dict, problem_id, {"run": "update ratio", "param": f"u_rate={ur}"})
    else:
        raise KeyError("Inputs are invalid.")


if __name__ == "__main__":
    # F18: fine-tuning
    # Population size
    reproduce_report(ga_f18_args_dict, 18, "p_size")
    # Mutation rate
    reproduce_report(ga_f18_args_dict, 18, "m_rate")
    # Update ratio
    reproduce_report(ga_f18_args_dict, 18, "u_rate")

    # F18: final results
    ga_main(ga_f18_args_dict, 18)

    # F19: fine-tuning
    # Population size
    reproduce_report(ga_f19_args_dict, 19, "p_size")
    # Mutation rate
    reproduce_report(ga_f19_args_dict, 19, "m_rate")
    # Update ratio
    reproduce_report(ga_f19_args_dict, 19, "u_rate")

    # F19: final results
    ga_main(ga_f19_args_dict, 19)
