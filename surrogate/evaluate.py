## @package evaluate
# @brief Test script for surrogate models.

# Read input arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default="test", help="Path to the data folder to delete.")
parser.add_argument("--model_path", type=str, default="models/model.keras", help="Path to save the trained model.")
parser.add_argument("--splitting_seed", type=int, default=40, help="Random seed for data splitting.")
parser.add_argument("--target", choices=["potential", "normal_derivative"], default="potential", help="Target quantity to predict.")
parser.add_argument("--prediction_seed", type=int, default=40, help="Random seed for test sample selection.")
args = parser.parse_args()
data_folder = args.folder
seed = args.splitting_seed
seed2 = args.prediction_seed
target = args.target
model_path = args.model_path

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 


# Import coordinates dataset and solutions
from .load import load
import numpy as np
mu, x, y, potential, x_plate, y_plate, normal_derivatives_plate = load(data_folder=data_folder)
if target == "potential":
    x = np.stack((x, y), axis=2)
    y = np.array(potential)
elif target == "normal_derivative":
    x = np.stack((x_plate, y_plate), axis=2)
    y = np.array(normal_derivatives_plate)

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

# Load the trained model
import tensorflow as tf
from .model import DenseNetwork, FourierFeatures, LogUniformFreqInitializer, EinsumLayer, DeepONet
from .losses import masked_mse, masked_mae
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

print("\033[38;2;0;175;6m\n\nEvaluating the model on the test set...\033[0m")
model.evaluate([mu_test, x_test], y_test)