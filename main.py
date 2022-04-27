import os, sys
sys.path.append(os.getcwd())

import numpy as np
import argparse

from pymoo.core.problem import Problem
from pymoo.factory import get_problem
from pymoo.optimize import minimize

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SimulatedBinaryCrossover
from pymoo.operators.mutation.pm import PolynomialMutation

from pymoo.util.plotting import plot
from pymoo.visualization.scatter import Scatter

from problem.problem import FON
from util import get_non_dominated_points
import plot.plotting_params as params


eta_c    = 10
eta_m    = 20
p_c      = 0.9
p_m      = 0.167

edgecolor= params.color_set1_pink/255.0
ms       = 20

# ==================================================================================================
# Main Starts here
# ==================================================================================================
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--num_gen', type=int, default=250,
                    help='number of generations (default: 64)')
parser.add_argument('--num_times', type=int, default=10,
                    help='num of times (default: 10)')
parser.add_argument('--pop_size', type=int, default=100,
                    help='population size (default: 100)')
parser.add_argument('--problem_name', type= str, default= 'FON',
                    help='problem to run on')
parser.add_argument('--dim', type= int, default= 2,
                    help='dimensions of the problem')
parser.add_argument('--init', action='store_true', default=False,
                    help='initialize with weighted sum')

args     = parser.parse_args()
num_gen  = args.num_gen
num_times= args.num_times
pop_size = args.pop_size
problem_name  = args.problem_name
dim      = args.dim

# Print the arguments


# Start executing main
if problem_name == "zdt4":
    problem  = get_problem("zdt4")
elif problem_name == "FON":
    problem  = FON()
# plot(problem.pareto_front(), no_fill=True)

algorithm= NSGA2(pop_size = pop_size,
                  crossover= SimulatedBinaryCrossover(eta= eta_c, prob= p_c),
                  mutation = PolynomialMutation      (eta= eta_m, prob= p_m))


for i in range(num_times):
    print("Running for seed {:2d}".format(i))
    res     = minimize(problem,
                       algorithm,
                       ('n_gen', num_gen),
                       seed= i,
                       verbose= False)

    # Combine all sets of solutions
    if i == 0:
        all_result = res.F
    else:
        all_result = np.vstack((all_result, res.F))

# Check non-dominated points
non_dominated = get_non_dominated_points(all_result, verbose= True)

np.save("output/non_dominated.npy", non_dominated)

plot = Scatter()
plot.add(non_dominated, facecolor= "None", edgecolor= edgecolor, s= ms)
plot.show()
