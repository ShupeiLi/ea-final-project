from ioh import get_problem, ProblemClass
from ioh import logger
import sys
import numpy as np
import time


# Declaration of problems to be tested.
# We obtain an interface of the OneMax problem here.
dimension = 50

"""
1 (fid) : The funciton ID of the problem in the problem suite. OneMax is 1 defined within the PBO class. 2 would correspond to another problem.
dimension : The dimension of the problem, which we have set to 50.
instance: In benchmarking libraries, problems often have multiple instances. These instances may vary slightly (e.g., different random noise, shifts, etc.) 
            to allow algorithms to be tested on a variety of conditions.
om(x) return the fitness value of 'x'
"""
om = get_problem(1, dimension=dimension, instance=1, problem_class=ProblemClass.PBO)
# We know the optimum of onemax
optimum = dimension

# Create default logger compatible with IOHanalyzer
# `root` indicates where the output files are stored.
# `folder_name` is the name of the folder containing all output. You should compress the folder 'run' and upload it to IOHanalyzer.
l = logger.Analyzer(root="data", 
    folder_name="run", 
    algorithm_name="genetic_algorithm", 
    algorithm_info="The lab session of the evolutionary algorithm course in LIACS")

om.attach_logger(l)


# Parameters setting
pop_size = ...
tournament_k = ...
mutation_rate = ...
crossover_probability = ...


# Uniform Crossover
def crossover(p1, p2):
    ...

# Standard bit mutation using mutation rate p
def mutation(p):
    ...

# Using the Fitness proportional selection
def mating_seletion(parent, parent_f) :    
    ...

def genetic_algorithm(func, budget = None):
    
    # budget of each run: 10000
    if budget is None:
        budget = 10000
    
    # f_opt : Optimal function value.
    # x_opt : Optimal solution.
    f_opt = sys.float_info.min
    x_opt = None
    
    # parent : A list that holds the binary strings representing potential solutions or individuals in the current population.
    # parent_f : A list that holds the fitness values corresponding to each individual in the parent list.
    parent = []
    parent_f = []
    for i in range(pop_size):

        # Initialization
        parent.append(np.random.randint(2, size = func.meta_data.n_variables))
        parent_f.append(func(parent[i]))
        budget = budget - 1

    while (f_opt < optimum and budget > 0):

        # Perform mating selection, crossover, and mutation to generate offspring
        offspring = ...
        
    # ioh function, to reset the recording status of the function.
    func.reset()
    print(f_opt,x_opt)
    return f_opt, x_opt

def main():
    # We run the algorithm 20 independent times.
    for _ in range(20):
        genetic_algorithm(om)

if __name__ == '__main__':
  start = time.time()
  main()
  end = time.time()
  print("The program takes %s seconds" % (end-start))
