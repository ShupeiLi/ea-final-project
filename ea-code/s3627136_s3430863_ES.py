from typing import Any
from ioh import get_problem
from ioh import logger, ProblemClass
import numpy as np
import scipy.stats as st
import time
import copy

BUDGET = 5000
DIMENSION = 50 
population_size = 20  # population size *
MU = 5  # Number of parents to select
LAMBDA = 20  # Number of offspring to generate *
RUNS = 20
SEED = 42
Mutate_StepSize = 0.5 # step size

# Step size

es_f18_args_dict = {"population_size": population_size, "lambda_": LAMBDA, "dim": DIMENSION, "budget": BUDGET,
                    "seed": SEED, "mu": MU, "stepSize" : Mutate_StepSize}
es_f19_args_dict = {"population_size": population_size, "lambda_": LAMBDA, "dim": DIMENSION, "budget": BUDGET,
                    "seed": SEED, "mu": MU, "stepSize" : Mutate_StepSize}


class ES():
    def __init__(self, population_size, dim, problem, seed, budget, stepSize, mu, lambda_):
        self.population_size = population_size
        self.dimension = dim
        self.problem = problem
        self.budget = budget
        self.fixed_seed = seed
        self.seed = self.fixed_seed
        self._sigma = stepSize
        self.tau =  1.0 / np.sqrt(self.problem.meta_data.n_variables)
        self.mu_ = mu # Control the number of parent
        self.lambda_ = lambda_ # Number of offspring
        self.step = 0

        # init population
        self.parents = []
        self.parents_sigma = []
        self.parents_fitness = []

        self.childs = []
        self.childs_sigma = []
        self.childs_fitness = []
    
    
    def initial_population(self):
        self.parents = np.random.normal(0, 1, size=(self.population_size, self.dimension))
        self.parents_sigma = [self._sigma] * self.population_size

    def mutate_oneSigma(self):
        for i in range(len(self.childs)):
            self.childs_sigma[i] = self.childs_sigma[i] * np.exp(np.random.normal(0, self.tau))
            for j in range(len(self.childs[i])):
                self.childs[i][j] = self.childs[i][j] + np.random.normal(0, self.childs_sigma[i])
    
    def gaussian_to_uniform(self, individual):
        ind_uniform = st.norm.cdf(individual)
        for i, element in enumerate(ind_uniform):
            ind_uniform[i] = 1 if element >= 0.5 else 0
        return ind_uniform.astype(int)
    
    def recombination(self):    
        for i in range(self.lambda_):
            [p1,p2] = np.random.choice(len(self.parents), 2, replace = False)
            offspring = (self.parents[p1] + self.parents[p2])/2
            sigma = (self.parents_sigma[p1] + self.parents_sigma[p2])/2    
            self.childs.append(offspring)    
            self.childs_sigma.append(sigma)
    
    def evaluation(self, pops):
        if self.step + len(pops) <= self.budget:
            self.step += len(pops)
            fitness = [self.problem(self.gaussian_to_uniform(ind)) for ind in pops]
        else:
            fitness = [self.problem(self.gaussian_to_uniform(ind)) for ind in pops[:self.budget-self.step]]
            self.step = self.budget
        return fitness
    
    def Selection(self):
        rank = np.argsort(self.childs_fitness)
        self.parents = [self.childs[i] for i in rank][-self.mu_:] 
        self.parents_sigma = [self.childs_sigma[i] for i in rank][-self.mu_:] 
        self.parents_fitness = [self.childs_fitness[i] for i in rank][-self.mu_:] 
        
    def clear_(self):
        self.parents = []
        self.parents_sigma = []
        self.parents_fitness = []
        self.childs = []
        self.childs_sigma = []
        self.childs_fitness = []
        self.step = 0
        self.fixed_seed += 1
        self.seed = self.fixed_seed

    
    def __call__(self):
        # Initialization
        np.random.seed(self.seed)
        self.initial_population()
        self.parents_fitness = self.evaluation(self.parents)

        # Loop
        while self.step < self.budget:   
            # Recombination
            self.recombination()
            # Mutation
            self.mutate_oneSigma()
            # Evaluation Fitness
            self.childs_fitness = self.evaluation(self.childs)
            # Selection
            self.Selection()

        self.problem.reset()
        self.clear_()


def es_main(es_args, problem_id, info=None):
    """Run the ES algorithm.
    :param ga_args: dict. Parameters defined in ES algorithm.
    :param problem_id: int. 18 or 19.
    :param info: dict. The information of the algorithm.
    """
    start = time.time()
    problem = get_problem(fid=problem_id, dimension=es_args["dim"], instance=1, problem_class=ProblemClass.PBO)
    es_args["problem"] = problem
    algorithm = ES(**es_args)
    if info is not None:
        es_logger = logger.Analyzer(root="data",
                                    folder_name=f"{info['run']} {info['param']} run",
                                    algorithm_name=f"s3627136_s3430863_ES: F{problem_id}, {info['param']}",
                                    algorithm_info=f"Practical assignment of the EA course."
                                    )
    else:
        es_logger = logger.Analyzer(root="data",
                                    folder_name="run",
                                    algorithm_name=f"s3627136_s3430863_ES: F{problem_id}",
                                    algorithm_info=f"Practical assignment of the EA course."
                                    )
    problem.attach_logger(es_logger)

    if info is not None:
        print(f"INFO: {info['param']}")
    for run in range(RUNS):
        print(f"Runs: {run + 1}")
        algorithm()


    es_logger.close()
    end = time.time()
    print("The program takes %s seconds" % (end - start))

def reproduce_report(args_dict, problem_id, param):
    """Reproduce fine-tuning results in our report."""
    if param == "p_size":
        pop_dict = copy.deepcopy(args_dict)
        for pop_k in [2, 5, 10, 20, 50]:
            pop_dict["population_size"] = pop_k
            es_main(pop_dict, problem_id, {"run": "population size", "param": f"p_size={pop_k}"})

    elif param == "s_size":
        step_dict = copy.deepcopy(args_dict)
        for ss in  [0.1, 0.3, 0.5, 0.7, 0.9]:
            step_dict["stepSize"] = ss
            es_main(step_dict, problem_id, {"run": "Step size", "param": f"s_size={ss}"})

    elif param == "o_size": # offspring size
        lbd_dict = copy.deepcopy(args_dict)
        for lbd in  [20, 40, 60, 80, 100]:
            lbd_dict["lambda_"] = lbd
            es_main(lbd_dict, problem_id, {"run": "Offspring size", "param": f"o_size={lbd}"})
    else:
        raise KeyError("Inputs are invalid.")
    

if __name__ == "__main__":
    # F18: fine-tuning
    # Population size
    reproduce_report(es_f18_args_dict, 18, "p_size")
    # Step size
    reproduce_report(es_f18_args_dict, 18, "s_size")
    # Offspring size
    reproduce_report(es_f18_args_dict, 18, "o_size")

    # F18: final results
    # es_main(es_f18_args_dict, 18)

    # F19: fine-tuning
    # Population size
    reproduce_report(es_f19_args_dict, 19, "p_size")
    # Step size
    reproduce_report(es_f19_args_dict, 19, "s_size")
    # Offspring size
    reproduce_report(es_f19_args_dict, 19, "o_size")

    # # F19: final results
    # es_main(es_f19_args_dict, 19)