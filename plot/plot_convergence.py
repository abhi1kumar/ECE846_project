import os, sys
sys.path.append(os.getcwd())

import numpy as np
from matplotlib import pyplot as plt
import plot.plotting_params as params

from lib.util import read_numpy
from plot.common_operations import savefig

def plot_per_problem(problem_name= "FON", dim_list = np.array([2, 4, 8, 16, 32])):
    np_data_folder = "output"
    save_folder   = "images"
    color_list = ['r', params.color_set1_pink/255.0, 'orange', 'purple', 'blue', 'k', params.color_set1_cyan/255.0]
    lw = params.lw-3
    legend_fs = params.legend_fs
    legend_border_axes_plot_border_pad = params.legend_border_axes_plot_border_pad
    legend_border_pad                  = params.legend_border_pad
    legend_vertical_label_spacing      = params.legend_vertical_label_spacing
    legend_marker_text_spacing         = params.legend_marker_text_spacing

    # ===================================
    # Plot for hypervolume
    # ===================================
    plt.figure(figsize= params.size, dpi= params.DPI)
    for i in range(dim_list.shape[0]):
        dim  = dim_list[i]

        np_data_path = os.path.join(np_data_folder, problem_name + "_" + str(dim) + "_random.npy" )
        data = read_numpy(path= np_data_path)
        iter = data[:, 0]
        hv   = data[:, 1]
        plt.plot(iter, hv, c= color_list[i], lw= lw, linestyle= 'dashed')

        np_data_path = os.path.join(np_data_folder, problem_name + "_" + str(dim) + "_weighted_sum.npy" )
        data = read_numpy(path= np_data_path)
        iter = data[:, 0]
        hv   = data[:, 1]
        plt.plot(iter, hv, c= color_list[i], lw= lw, linestyle= 'solid', label= str(dim))
    plt.xlim(1, data.shape[0])
    plt.ylim(bottom= 0)
    plt.xlabel('Iteration')
    plt.ylabel('HyperVolume')
    plt.grid(True)
    plt.legend(loc= 'lower right', fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
    save_path = os.path.join(save_folder, problem_name + "_hv.png")
    savefig(plt, path= save_path)

    # ===================================
    # Plot for IGD
    # ===================================
    plt.figure(figsize= params.size, dpi= params.DPI)
    for i in range(dim_list.shape[0]):
        dim  = dim_list[i]

        np_data_path = os.path.join(np_data_folder, problem_name + "_" + str(dim) + "_random.npy" )
        data = read_numpy(path= np_data_path)
        iter = data[:, 0]
        igd  = data[:, 2]
        plt.plot(iter, igd, c= color_list[i], lw= lw, linestyle= 'dashed')

        np_data_path = os.path.join(np_data_folder, problem_name + "_" + str(dim) + "_weighted_sum.npy" )
        data = read_numpy(path= np_data_path)
        iter = data[:, 0]
        igd   = data[:, 2]
        plt.plot(iter, igd, c= color_list[i], lw= lw, linestyle= 'solid', label= str(dim))
    plt.xlim(1, data.shape[0])
    plt.ylim(bottom= 0)
    plt.xlabel('Iteration')
    plt.ylabel('IGD')
    plt.grid(True)
    plt.legend(loc= 'upper right', fontsize= legend_fs, borderaxespad= legend_border_axes_plot_border_pad, borderpad= legend_border_pad, labelspacing= legend_vertical_label_spacing, handletextpad= legend_marker_text_spacing)
    save_path = os.path.join(save_folder, problem_name + "_igd.png")
    savefig(plt, path= save_path)


# ==================================================================================================
# Main Starts here
# ==================================================================================================
plot_per_problem(problem_name= "FON", dim_list = np.array([2, 4, 8, 16, 32, 64]))
plot_per_problem(problem_name= "ZDT1", dim_list = np.array([2, 4, 8, 16, 30]))