import os, sys
sys.path.append(os.getcwd())

import numpy as np
import scipy.io as sio
from scipy.spatial import distance

def read_dat(file_path):
    with open(file_path) as f:
        data_str = [line.strip("\n").split("\t") for line in f]
    return np.array(data_str).astype(np.float)

def read_mat(file_path):
    print("Loading {}".format(file_path))
    mat_contents = sio.loadmat(file_path)

    return mat_contents


def get_non_dominated_points(func_data, x_data, verbose= False):
    if verbose:
        print("\nRemoving non-dominated points...")
        print("Before removal shape", func_data.shape)

    N, D = func_data.shape
    dom_index = []
    for i in range(N):
        temp_diff = func_data - func_data[i]
        # Find how many of them are negative
        temp_indicator     = np.sign(temp_diff)
        temp_indicator_sum = np.sum (temp_indicator, axis= 1).astype(int)

        # The indices which have exact D -1 are dominated   
        temp_index = np.where(temp_indicator_sum == -D)[0]

        # No-dominated index
        if temp_index.shape[0] == 0:
            continue

        if len(dom_index) == 0:
            dom_index = temp_index
        else:
            dom_index = np.hstack((dom_index, temp_index))

    # Filter out the dominated indices
    good_index = np.setdiff1d(np.arange(N), dom_index)

    if verbose:
        print("After removal shape", func_data[good_index].shape)

    return func_data[good_index], x_data[good_index]

def get_closest_index(data, K, INF= np.inf):
    dist_matrix = distance.cdist(data, data, 'euclidean')

    # Assign diagonals to be infinity so that they are not counted as neighbours
    np.fill_diagonal(dist_matrix, INF)

    # Find K min in each row.
    indices_argsort = np.argsort(dist_matrix , axis = -1)[:, :K]

    return indices_argsort

def get_tradeoff_curr_point(data, closest_index, curr_index):

    diff_for_loss = data[closest_index] - data[curr_index]
    avg_loss      = np.maximum(diff_for_loss, 0).sum(axis=1) / np.maximum(np.sign(diff_for_loss), 0).sum(axis=1)
    diff_for_gain = - diff_for_loss
    avg_gain      = np.maximum(diff_for_gain, 0).sum(axis=1) / np.maximum(np.sign(diff_for_loss), 0).sum(axis=1)

    tradeoff_curr_point = np.max(avg_loss /avg_gain)

    return tradeoff_curr_point

def get_tradeoff(data, closest_index):
    N, dim = data.shape
    tradeoff = np.zeros((N, ))

    for i in range(N):
        tradeoff[i] = get_tradeoff_curr_point(data= data, closest_index= closest_index[i], curr_index= i)

    return tradeoff
