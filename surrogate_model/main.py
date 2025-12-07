## @file main.py
# @brief Main script to train and evaluate surrogate models for geometry-to-solution mapping.

import numpy as np



from data.plot import summary_plot
import h5py
path = data_folder + f"/results/1.h5"
with h5py.File(path, "r") as file:
    summary_plot(file)



