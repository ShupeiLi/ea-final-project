from typing import Any
from ioh import get_problem
from ioh import logger, ProblemClass
import numpy as np
import scipy.stats as st
import os
import shutil
import time
import copy

BUDGET = 5000
DIMENSION = 50 
population_size = 20  
LAMBDA_RATE = 2 
RUNS = 20
SEED = 42
Mutate_StepSize = 0.5

np.random.seed(SEED)
# Step size

es_f18_args_dict = {"population_size": population_size, "lambda_rate": LAMBDA_RATE, "dim": DIMENSION, "budget": BUDGET,
                    "seed": SEED, "stepSize" : Mutate_StepSize}
es_f19_args_dict = {"population_size": population_size, "lambda_rate": LAMBDA_RATE, "dim": DIMENSION, "budget": BUDGET,
                    "seed": SEED, "stepSize" : Mutate_StepSize}


class ES():
    def __init__(self, population_size, dim, problem, budget, stepSize, lambda_rate, seed):
        self.population_size = population_size # mu_
        self.dimension = dim
        self.problem = problem
        self.budget = budget
        self._sigma = stepSize
        self.tau =  1.0 / np.sqrt(self.problem.meta_data.n_variables)
        self.lambda_ = int(lambda_rate * population_size) # Number of offspring
        self.step = 0
        self.optfitness = 0

        # init population
        self.parents = []
        self.parents_sigma = []
        self.parents_fitness = []

        self.childs = []
        self.childs_sigma = []
        self.childs_fitness = []

        self.population = []
        self.population_sigma = []
        self.population_fitness = []
    
    
    def initial_population(self):
        self.parents = np.random.normal(0, 1, size=(self.population_size, self.dimension))
        self.parents_sigma = [self._sigma] * self.population_size
        self.parents_fitness = self.evaluation(self.parents)

    def mutate_oneSigma(self):
        # mutate each individual in offspring
        for i in range(len(self.childs)):
            self.childs_sigma[i] = self.childs_sigma[i] * np.exp(np.random.normal(0, self.tau))
            for j in range(len(self.childs[i])):
                self.childs[i][j] = self.childs[i][j] + np.random.normal(0, self.childs_sigma[i])
        self.childs_fitness = self.evaluation(self.childs)
    
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
        self.population = np.concatenate((self.parents, self.childs), axis=0)
        self.population_sigma = np.concatenate((self.parents_sigma, self.childs_sigma), axis=0)
        self.population_fitness = np.concatenate((self.parents_fitness, self.childs_fitness), axis=0)

        # Next generation Parents
        rank = np.argsort(self.population_fitness)
        self.parents = [self.population[i] for i in rank][-self.population_size:] 
        self.parents_sigma = [self.population_sigma[i] for i in rank][-self.population_size:] 
        self.parents_fitness = [self.population_fitness[i] for i in rank][-self.population_size:] 
        if self.parents_fitness[-1] > self.optfitness:
            self.optfitness = self.parents_fitness[-1]
        
    def clear_(self):
        self.childs = []
        self.childs_sigma = []
        self.childs_fitness = []
        self.population = []
        self.population_sigma = []
        self.population_fitness = []

    def reset(self):
        self.clear_()
        self.parents = []
        self.parents_sigma = []
        self.parents_fitness = []
        self.problem.reset()
        self.step = 0
        self.optfitness = 0
    
    def __call__(self):
        # Initialization parents, parents_sigma => evaluate parents
        self.initial_population()
        # print(self.parents[0])

        # Loop
        while self.step < self.budget:   
            # Recombination to generate lambda size offspring (lambda_ > populationsize(mu_))
            self.recombination()
            # Mutation on all individuals in offspring => evaluate childs
            self.mutate_oneSigma()
            # Selection on (partents + childs) => next generation parents, with sigma, fitness
            self.Selection()
            # clear childs
            self.clear_()
        print(self.optfitness)
        self.reset()
    


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
        problem.reset()
        algorithm.reset()


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

    elif param == "lbd_rate": # offspring size
        lbd_dict = copy.deepcopy(args_dict)
        if problem_id == 18:
            lbd_dict["population_size"] = 10
            lbd_dict["stepSize"] = 0.3
        if problem_id == 19:
            lbd_dict["population_size"] = 50
            lbd_dict["stepSize"] = 0.7
        for lbd in  [1.5, 2, 2.5, 5, 10]:
            lbd_dict["lambda_rate"] = lbd
            es_main(lbd_dict, problem_id, {"run": "Offspring size", "param": f"lbd_rate={lbd}"})
    else:
        raise KeyError("Inputs are invalid.")

def final_results():
    path = "./data"
    F18_best_result = "population size p_size=10 run" 
    F19_best_result = "Step size s_size=0.7 run-1"
    try:
        folders = [f.path for f in os.scandir(path) if f.is_dir()]
        for folder in folders:
            print(os.path.basename(folder))
            folder_name = os.path.basename(folder)
            if folder_name != F18_best_result and folder_name != F19_best_result :
                shutil.rmtree(folder)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # F18: fine-tuning
    # Population size
    reproduce_report(es_f18_args_dict, 18, "p_size")
    # Step size
    reproduce_report(es_f18_args_dict, 18, "s_size")
    # Offspring size
    reproduce_report(es_f18_args_dict, 18, "lbd_rate")

    # F19: fine-tuning
    # Population size
    reproduce_report(es_f19_args_dict, 19, "p_size")
    # Step size
    reproduce_report(es_f19_args_dict, 19, "s_size")
    # Offspring size
    # reproduce_report(es_f19_args_dict, 19, "lbd_rate")

    # Best result
    final_results()

    