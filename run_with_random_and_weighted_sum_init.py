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
from plot.common_operations import *
import plot.plotting_params as params

eta_c    = 10
eta_m    = 20
p_c      = 0.9
p_m      = 0.167

legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
legend_border_pad                  = params.legend_border_pad
legend_vertical_label_spacing      = params.legend_vertical_label_spacing
legend_marker_text_spacing         = params.legend_marker_text_spacing

color1   = params.color_set1_pink/255.0
color2   = params.color_set1_cyan/255.0
color_gt = 'limegreen'
ms       = 50

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

def run_nsga_num_steps(problem, pareto_X, pop_size, num_gen, do_init= False, init_solution= None, edgecolor= color1, save_gif= True, gif_path= None):
    hv  = get_performance_indicator("hv", ref_point=np.array([1.2, 1.2]))
    igd = get_performance_indicator("igd", pareto_X)
    pareto_F = problem.evaluate(pareto_X)

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

    if save_gif:
        gif_writer = open_gif_writer(gif_path)

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

        if save_gif:
            if i % 1 == 0 or i == num_gen-1:
                t = problem.evaluate(sol_non_dominated)
                fig = plt.figure(figsize= params.size, dpi= params.DPI)
                plt.scatter(pareto_F[:, 0], pareto_F[:, 1], color= color_gt, s= ms-5, label= 'GT')
                plt.scatter(t[:,0], t[:,1], color= edgecolor, s= ms, label= 'Found')
                plt.xlabel(r'$f_1$')
                plt.ylabel(r'$f_2$')
                plt.grid(True)
                plt.xlim(0, xmax)
                plt.ylim(0, ymax)
                plt.legend(loc= 'upper right', borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
                plt.title('Iteration ' + str(i+1))
                add_ubyte_image_to_gif_writer(gif_writer, convert_fig_to_ubyte_image(fig))
                plt.close()

    close_gif_writer(gif_writer)


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
parser.add_argument('--save_gif', action='store_true', default=False,
                        help='save GIF')
args     = parser.parse_args()
problem_name  = args.problem_name
dim      = args.dim
num_gen  = args.num_gen
num_seeds= args.num_seeds
pop_size = args.pop_size
save_gif = args.save_gif

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
    pareto_X       = np.zeros((temp.shape[0], dim))
    pareto_X[:, 0] = temp
    xmax     = 1
    ymax     = 4

elif problem_name == "ZDT4":
    problem  = ZDT4(n_var= dim)
    init_solution   = read_mat("matlab/ZDT4_" + str(dim) + '.mat')['x_init_matrix']
    temp     = np.arange(0, 1, 0.01)
    pareto_X       = np.zeros((temp.shape[0], dim))
    pareto_X[:, 0] = temp
    xmax     = 1
    ymax     = 4


elif problem_name == "FON":
    problem  = FON(n_var= dim)
    init_solution   = read_mat("matlab/FON_" + str(dim) + '.mat')['x_init_matrix']
    temp = np.arange(-1.0/math.sqrt(dim), 1.0/math.sqrt(dim), 0.01)
    pareto_X = np.repeat(temp[:, np.newaxis], dim, axis=1)
    xmax     = 1
    ymax     = 1


print(init_solution.shape)
print(pareto_X.shape)

num_init_sol = init_solution.shape[0]
if pop_size < num_init_sol:
    # downsample
    delta = num_init_sol // pop_size
    init_solution = init_solution[::delta]

# Run without initialization
gif_path = os.path.join("images", problem_name + "_" + str(dim) + "_random.gif")
run_nsga_num_steps(problem, pareto_X, pop_size, num_gen, do_init= False, init_solution= None, edgecolor= color2, save_gif= save_gif, gif_path= gif_path)

# Run with initialization
gif_path = os.path.join("images", problem_name + "_" + str(dim) + "_weighted_sum.gif")
run_nsga_num_steps(problem, pareto_X, pop_size, num_gen, do_init= True, init_solution= init_solution, edgecolor= color1, save_gif= save_gif, gif_path= gif_path)
