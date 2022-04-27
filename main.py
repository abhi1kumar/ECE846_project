import os, sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import copy
import math

from pymoo.core.problem import Problem
from pymoo.factory import get_problem, get_performance_indicator
from pymoo.optimize import minimize

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from pymoo.util.plotting import plot
from pymoo.visualization.scatter import Scatter

from lib.problem import FON
from lib.util import get_non_dominated_points, read_mat
import plot.plotting_params as params

from pymoo.config import Config
Config.show_compile_hint = False

eta_c    = 10
eta_m    = 20
p_c      = 0.9
p_m      = 0.167

edgecolor= params.color_set1_pink/255.0
ms       = 20

def run_nsga_one_step(problem, pop_size, do_init= False, init_solution= None):

    for i in range(num_times):
        # print("Running for seed {:2d}".format(i))

        if do_init:
            if init_solution.ndim == 3:
                init_data_curr = init_solution[i]
            elif init_solution.ndim == 2:
                init_data_curr = init_solution
            else:
                raise NotImplementedError

            algorithm= NSGA2(pop_size = pop_size,
                          crossover= SimulatedBinaryCrossover(eta= eta_c, prob= p_c),
                          mutation = PolynomialMutation      (eta= eta_m, prob= p_m),
                          sampling = init_data_curr)
        else:
            algorithm= NSGA2(pop_size = pop_size,
                          crossover= SimulatedBinaryCrossover(eta= eta_c, prob= p_c),
                          mutation = PolynomialMutation      (eta= eta_m, prob= p_m))

        res     = minimize(problem,
                           algorithm,
                           ('n_gen', 1),
                           seed= i,
                           verbose= False)

        sol = res.F
        # Combine all sets of solutions
        if i == 0:
            sol_all_seed = np.zeros((num_times, sol.shape[0], sol.shape[1]))
        sol_all_seed[i] = sol

    # Check non-dominated points
    sol_non_dominated = get_non_dominated_points(sol_all_seed.reshape(-1, sol.shape[1]), verbose= False)

    return sol_all_seed, sol_non_dominated


def run_nsga_num_steps(problem, pareto_solution, pop_size, num_gen, do_init= False, init_solution= None):
    hv  = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
    igd = get_performance_indicator("igd", pareto_solution)

    non_dominated_all_step = []
    for j in range(num_gen):
        if j == 0:
            if do_init:
                print("\nNSGA2 with weighted sum initialization...")
            else:
                print("\nNSGA2 with random initialization...")

            sol_all_seed, sol_non_dominated = run_nsga_one_step(problem, pop_size, do_init= do_init, init_solution= init_solution)
        else:
            sol_all_seed, sol_non_dominated = run_nsga_one_step(problem, pop_size, do_init= do_init, init_solution= sol_all_seed)

        # Print some statistics
        hv_val  = igd.do(sol_non_dominated)
        igd_val = hv.do(sol_non_dominated)
        print("Gen= {:2d} HV= {:.2f} IGD= {:.2f}".format(j+1, hv_val, igd_val))

        non_dominated_all_step.append(sol_non_dominated)

# ==================================================================================================
# Main Starts here
# ==================================================================================================
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--problem_name', type= str, default= 'FON',
                    help='problem to run on')
parser.add_argument('--dim', type= int, default= 2,
                    help='dimensions of the problem')

parser.add_argument('--num_gen', type=int, default=5,
                    help='number of generations (default: 20)')
parser.add_argument('--num_times', type=int, default=10,
                    help='num of times (default: 10)')
parser.add_argument('--pop_size', type=int, default=100,
                    help='population size (default: 100)')
args     = parser.parse_args()
problem_name  = args.problem_name
dim      = args.dim
num_gen  = args.num_gen
num_times= args.num_times
pop_size = args.pop_size

# Print the arguments
print("Problem  = {}".format(problem_name))
print("Dim      = {}".format(dim))
print("Num_gen  = {}".format(num_gen))
print("Num times= {}".format(num_times))
print("Pop size = {}".format(pop_size))

# Start executing main
if problem_name == "zdt4":
    problem  = get_problem("zdt4")
    pareto_solution  = problem.pareto_front()

elif problem_name == "FON":
    problem  = FON()
    init_solution   = read_mat("matlab/FON_" + str(dim) + '.mat')['x_init_matrix']
    temp = np.arange(-1.0/math.sqrt(dim), 1.0/math.sqrt(dim), 0.01)
    pareto_solution = np.repeat(temp[:, np.newaxis], dim, axis=1)

num_init_sol = init_solution.shape[0]
if pop_size < num_init_sol:
    # downsample
    delta = num_init_sol // pop_size
    init_solution = init_solution[::delta]

# print(pareto_solution.shape)

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
