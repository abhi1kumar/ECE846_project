import os, sys
sys.path.append(os.getcwd())

import argparse
import numpy as np
import copy
import math

from pymoo.factory import get_performance_indicator
from pymoo.optimize import minimize
from pymoo.problems.multi import ZDT4

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

def list_to_append_array(mylist):
    num_times = len(mylist)
    appended_arr = mylist[0]
    for i in range(1, num_times):
        appended_arr = np.vstack((appended_arr, mylist[i]))

    return appended_arr

def run_nsga_one_step(problem, pop_size, do_init= False, init_solution= None):

    for i in range(num_times):
        # print("Running for seed {:2d}".format(i))

        if do_init:
            # Reference
            # https://pymoo.org/customization/initialization.html?highlight=population%20nsga2
            if type(init_solution) == np.ndarray and init_solution.ndim == 2:
                init_data_curr = init_solution
            elif isinstance(init_solution, list):
                init_data_curr = init_solution[i]
            else:
                raise NotImplementedError

            algorithm= NSGA2(pop_size = pop_size,
                          crossover= SimulatedBinaryCrossover(eta= eta_c, prob= p_c),
                          mutation = PolynomialMutation      (eta= eta_m, prob= p_m),
                          eliminate_duplicates=True,
                          sampling = init_data_curr)
        else:
            algorithm= NSGA2(pop_size = pop_size,
                          crossover= SimulatedBinaryCrossover(eta= eta_c, prob= p_c),
                          mutation = PolynomialMutation      (eta= eta_m, prob= p_m),
                          eliminate_duplicates=True)

        res     = minimize(problem,
                           algorithm,
                           ('n_gen', 1),
                           seed= i,
                           verbose= False)

        func = res.F
        sol  = res.X

        # Combine all sets of solutions
        if i == 0:
            func_all_seed = []#np.zeros((num_times, func.shape[0], func.shape[1]))
            sol_all_seed  = []#np.zeros((num_times, sol.shape[0], sol.shape[1]))
        func_all_seed.append(func)
        sol_all_seed.append(sol)

    func_all_seed_appended = list_to_append_array(func_all_seed)
    sol_all_seed_appended  = list_to_append_array(sol_all_seed)

    # Get non-dominated points
    func_non_dominated, sol_non_dominated = get_non_dominated_points(func_all_seed_appended, sol_all_seed_appended, verbose= False)

    return func_all_seed, sol_all_seed, func_non_dominated, sol_non_dominated


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

            func_all_seed, sol_all_seed, func_non_dominated, sol_non_dominated = run_nsga_one_step(problem, pop_size, do_init= do_init, init_solution= init_solution)
        else:
            func_all_seed, sol_all_seed, func_non_dominated, sol_non_dominated = run_nsga_one_step(problem, pop_size, do_init= do_init, init_solution= sol_all_seed)

        # Print some statistics
        hv_val  = hv.do(func_non_dominated)
        igd_val = igd.do(sol_non_dominated)
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
print("==================================")
print("Problem  = {}".format(problem_name))
print("Dim      = {}".format(dim))
print("Num_gen  = {}".format(num_gen))
print("Num times= {}".format(num_times))
print("Pop size = {}".format(pop_size))
print("==================================\n")

# Start executing main
if problem_name == "ZDT4":
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
