#@file src/meshes_coordinates.py
#@brief Script to extract and save coordinates from h5 files to get one file
# for each parameter that contains the values from all of the meshes.

import h5py
import numpy as np

def load_h5_solutions():
    """Loads solution data from multiple HDF5 files, pads them to ensure uniform length,
    and returns the stacked arrays along with a mask.
    """
    # Empty lists to collect data from all files
    all_x = []
    all_y = []
    all_potential = []
    all_grad_x = []
    all_grad_y = []

    for i in range(1, 1001):
        path = f"data/results/{i}_solution.h5"
        with h5py.File(path, 'r') as file:
            all_x.append(file['coord_x'][:])
            all_y.append(file['coord_y'][:])
            all_potential.append(file['potential'][:])
            all_grad_x.append(file['grad_x'][:])
            all_grad_y.append(file['grad_y'][:])

    # Find maximum length across all files
    max_len = np.max([len(arr) for arr in all_x])

    # Save lengths of each array before padding
    lengths = [len(arr) for arr in all_x]   # before padding


    # Pad arrays with 0s to ensure uniform length
    for i in range(len(all_x)):
        all_x[i] = np.pad(all_x[i], (0, max_len - len(all_x[i])), 'constant', constant_values=0)
        all_y[i] = np.pad(all_y[i], (0, max_len - len(all_y[i])), 'constant', constant_values=0)
        all_potential[i] = np.pad(all_potential[i], (0, max_len - len(all_potential[i])), 'constant', constant_values=np.nan)
        all_grad_x[i] = np.pad(all_grad_x[i], (0, max_len - len(all_grad_x[i])), 'constant', constant_values=np.nan)
        all_grad_y[i] = np.pad(all_grad_y[i], (0, max_len - len(all_grad_y[i])), 'constant', constant_values=np.nan)

    # Stack arrays in each list to create 2D matrixes
    x_matrix = np.stack(all_x)
    y_matrix = np.stack(all_y)
    potential_matrix = np.stack(all_potential)
    grad_x_matrix = np.stack(all_grad_x)
    grad_y_matrix = np.stack(all_grad_y)

    return x_matrix, y_matrix, potential_matrix, grad_x_matrix, grad_y_matrix