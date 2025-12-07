## @package predict
# @brief Functions to test surrogate models for geometry-to-solution mapping.

data_folder = "test"
seed = 44

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# Import coordinates dataset and solutions
from .load import load
import numpy as np
mu, x, y, potential, x_plate, y_plate, normal_derivatives_plate = load(data_folder=data_folder)
x = np.stack((x, y), axis=2)
y = np.array(potential)

# Split the dataset into training, validation, and test sets as in the training script
from sklearn.model_selection import train_test_split
ns = mu.shape[0]
idx = np.arange(ns)
idx_trainval, idx_test = train_test_split(idx, test_size=0.2, random_state=seed)
# Split train+val indices into train and val sets
idx_train, idx_val = train_test_split(idx_trainval, test_size=0.2, random_state=seed)
# Use indices to split the arrays along the first dimension
mu_train, mu_val, mu_test = mu[idx_train], mu[idx_val], mu[idx_test]
x_train, x_val, x_test = x[idx_train], x[idx_val], x[idx_test]
y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

# Identify the index of a random test sample
import random
random.seed(seed)
test_sample_index = random.choice(idx_test)
# Extract the corresponding input and output data
x_sample = x[test_sample_index:test_sample_index+1]
y_sample = y[test_sample_index:test_sample_index+1]
mu_sample = mu[test_sample_index:test_sample_index+1]

# Load the trained model
import tensorflow as tf
from .model import DenseNetwork, FourierFeatures, LogUniformFreqInitializer, EinsumLayer, DeepONet
from .losses import masked_mse, masked_mae
model_path = "models/model.keras"
model = tf.keras.models.load_model(
    model_path, 
    custom_objects={
        "DenseNetwork": DenseNetwork, 
        'FourierFeatures': FourierFeatures, 
        'LogUniformFreqInitializer': LogUniformFreqInitializer, 
        'EinsumLayer': EinsumLayer, 
        'DeepONet': DeepONet,
        'masked_mse': masked_mse,
        'masked_mae': masked_mae,
        })
print("\033[38;2;0;175;6m\n\nLoaded surrogate model summary.\033[0m")
model.summary()

# Make prediction for the selected test sample
# Print some information
print("\033[38;2;0;175;6m\n\nTesting the surrogate model on a random test sample.\033[0m")
y_pred = model([mu_sample,x_sample]).numpy()
print("Prediction shape:   ", y_pred.shape)
branch_output = model.call([mu_sample, x_sample], return_branch=True).numpy()
print("Branch output shape:", branch_output.shape)
trunk_output = model.call([mu_sample, x_sample], return_trunk=True).numpy()
print("Trunk output shape: ", trunk_output.shape)

# Save y_pred to a .h5 file with the same structure as the FOM results
# Find the number of the file corresponding to the test sample
# Open parameters.csv to find the index of the test sample that has the corresponding mu
import csv

def find_id(mu: np.ndarray, 
            data_folder: str = "data") -> int:
    parameters_file = os.path.join(data_folder, "parameters.csv")
    with open(parameters_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header
        mu_list = []
        for row in reader:
            mu_values = [float(value) for value in row]
            # Remove the first element since it's the index
            mu_values = mu_values[1:]
            mu_list.append(mu_values)
    mu_array = np.array(mu_list)
    # Find the index of the test sample in mu_array
    test_sample_number = None
    for i in range(mu_array.shape[0]):
        if np.allclose(mu_array[i], mu):
            test_sample_number = i+1
            break
    if test_sample_number is None:
        raise ValueError("Test sample mu not found in parameters.csv")
    return test_sample_number

idx = find_id(mu=mu_sample[0], data_folder=data_folder)


# Read the corresponding .h5 file in data_folder/results and plot the error
print(f"\033[38;2;0;175;6m\n\nSaving prediction and plotting results for test sample index {idx}.\033[0m")
print(f"Results file: {os.path.join(data_folder, 'results', f'{idx}.h5')}")
print(f"\033[38;2;0;175;6m\n\nPlotting prediction and error.\033[0m")
print("\n\n")
import h5py
from data.plot import plot_error, plot_potential, plot_potential_pred
import matplotlib.pyplot as plt
fom_file = os.path.join(data_folder, "results", f"{idx}.h5")
with h5py.File(fom_file, 'a') as file:
    # use x[0] to dectect and remove nan values in y_pred
    nan_mask = ~np.isnan(y_sample[0])
    if "potential_pred" in file:
        del file["potential_pred"]
    file["potential_pred"] = y_pred[0][nan_mask]
    if "se" in file:
        del file["se"]
    file["se"] = (y_pred[0][nan_mask] - file["potential"][:])**2
    if "ae" in file:
        del file["ae"]
    file["ae"] = np.abs(y_pred[0][nan_mask] - file["potential"][:])
    plot_potential(file, postpone_show=True)
    plot_error(file, postpone_show=True)
    plot_potential_pred(file, postpone_show=True)
    plt.show()