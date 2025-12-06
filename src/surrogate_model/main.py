## @file main.py
# @brief Main script to train and evaluate surrogate models for geometry-to-solution mapping.

import numpy as np

# Import coordinates dataset and solutions
from load import load
data_folder = "test"
mu, x, y, potential, x_plate, y_plate, normal_derivatives_plate = load(data_folder=data_folder)

# Define input and output arrays
x = np.stack((x, y), axis=2)
y = np.array(potential) 

# Train the model
from gpu import run_on_device
from train import train
run_on_device(train, model_path="models/model.keras", r=20, x=x, y=y, mu=mu)

