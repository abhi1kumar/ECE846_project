import os, sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import copy
import math
from matplotlib import pyplot as plt

from pymoo.factory import get_performance_indicator
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT1, ZDT4

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from pymoo.core.callback import Callback
from pymoo.util.plotting import plot
from pymoo.visualization.scatter import Scatter

# Suppress warnings
from pymoo.config import Config
Config.show_compile_hint = False

from lib.problem import FON
from lib.util import get_non_dominated_points, read_mat, list_to_append_array
import plot.plotting_params as params

eta_c    = 10
eta_m    = 20
p_c      = 0.9
p_m      = 0.167

edgecolor= params.color_set1_pink/255.0
ms       = 20

class MyCallback(Callback):

    # Reference
    # https://pymoo.org/interface/callback.html?highlight=callback
    def __init__(self) -> None:
        super().__init__()
        self.data["F"] = []
        self.data["X"] = []

    def notify(self, algorithm):
        self.data["F"].append(algorithm.pop.get("F"))
        self.data["X"].append(algorithm.pop.get("X"))

def run_nsga_num_steps(problem, pareto_solution, pop_size, num_gen, do_init= False, init_solution= None):
    hv  = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
    igd = get_performance_indicator("igd", pareto_solution)

    if do_init:
        print("\nWeighted-sum Initialization...")
        algorithm= NSGA2(pop_size = pop_size,
                     crossover= SimulatedBinaryCrossover(eta= eta_c, prob= p_c),
                     mutation = PolynomialMutation      (eta= eta_m, prob= p_m),
                     eliminate_duplicates=True,
                     sampling = init_solution
                     )
    else:
        print("\nRandom Initialization...")
        algorithm= NSGA2(pop_size = pop_size,
                         crossover= SimulatedBinaryCrossover(eta= eta_c, prob= p_c),
                         mutation = PolynomialMutation      (eta= eta_m, prob= p_m),
                         eliminate_duplicates=True
                         )

    func_all_seed_all_gen = []
    sol_all_seed_all_gen = []
    for i in range(num_seeds):
        res      = minimize(problem,
                            algorithm,
                            ('n_gen', num_gen),
                            seed= i,
                            callback=MyCallback(),
                            verbose= False)
        func_all_seed_all_gen.append(res.algorithm.callback.data["F"])
        sol_all_seed_all_gen.append(res.algorithm.callback.data["X"])

    for i in range(num_gen):
        func_all_seed = []
        sol_all_seed  = []
        for j in range(num_seeds):
            func_all_seed.append(func_all_seed_all_gen[j][i])
            sol_all_seed.append (sol_all_seed_all_gen [j][i])

        func_all_seed_appended = list_to_append_array(func_all_seed)
        sol_all_seed_appended  = list_to_append_array(sol_all_seed)

        # Get non-dominated points
        func_non_dominated, sol_non_dominated = get_non_dominated_points(func_all_seed_appended, sol_all_seed_appended, verbose= False)

        # Print some statistics
        hv_val  = hv.do(func_non_dominated)
        igd_val = igd.do(sol_non_dominated)
        print("Gen= {:2d} HV= {:.2f} IGD= {:.2f}".format(i+1, hv_val, igd_val))

        if i % 10 == 0 or i == num_gen-1:
            t = problem.evaluate(sol_non_dominated)
            plt.scatter(t[:,0], t[:,1]);
            plt.show()
            plt.close()

# ==================================================================================================
# Main Starts here
# ==================================================================================================
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--problem_name', type= str, default= 'FON',
                    help='problem to run on')
parser.add_argument('--dim', type= int, default= 2,
                    help='dimensions of the problem')

parser.add_argument('--num_gen', type=int, default=50,
                    help='number of generations (default: 50)')
parser.add_argument('--num_seeds', type=int, default=5,
                    help='num of seeds (default: 5)')
parser.add_argument('--pop_size', type=int, default=100,
                    help='population size (default: 100)')
args     = parser.parse_args()
problem_name  = args.problem_name
dim      = args.dim
num_gen  = args.num_gen
num_seeds= args.num_seeds
pop_size = args.pop_size

# Print the arguments
print("==================================")
print("Problem  = {}".format(problem_name))
print("Dim      = {}".format(dim))
print("Num_gen  = {}".format(num_gen))
print("Num seeds= {}".format(num_seeds))
print("Pop size = {}".format(pop_size))
print("==================================\n")

# Start executing main
if problem_name == "ZDT1":
    problem  = ZDT1(n_var= dim)
    init_solution   = read_mat("matlab/ZDT1_" + str(dim) + '.mat')['x_init_matrix']
    temp     = np.arange(0, 1, 0.01)
    pareto_solution       = np.zeros((temp.shape[0], dim))
    pareto_solution[:, 0] = temp

elif problem_name == "ZDT4":
    problem  = ZDT4(n_var= dim)
    init_solution   = read_mat("matlab/ZDT4_" + str(dim) + '.mat')['x_init_matrix']
    temp     = np.arange(0, 1, 0.01)
    pareto_solution       = np.zeros((temp.shape[0], dim))
    pareto_solution[:, 0] = temp


elif problem_name == "FON":
    problem  = FON(n_var= dim)
    init_solution   = read_mat("matlab/FON_" + str(dim) + '.mat')['x_init_matrix']
    temp = np.arange(-1.0/math.sqrt(dim), 1.0/math.sqrt(dim), 0.01)
    pareto_solution = np.repeat(temp[:, np.newaxis], dim, axis=1)

print(init_solution.shape)
print(pareto_solution.shape)

num_init_sol = init_solution.shape[0]
if pop_size < num_init_sol:
    # downsample
    delta = num_init_sol // pop_size
    init_solution = init_solution[::delta]

# Run without initialization
run_nsga_num_steps(problem, pareto_solution, pop_size, num_gen, do_init= False, init_solution= None)

# Run with initialization
run_nsga_num_steps(problem, pareto_solution, pop_size, num_gen, do_init= True , init_solution= init_solution)

# plot(problem.pareto_front(), no_fill=True)




#
# np.save("output/non_dominated.npy", non_dominated)
#
# plot = Scatter()
# plot.add(non_dominated, facecolor= "None", edgecolor= edgecolor, s= ms)
# plot.show()
