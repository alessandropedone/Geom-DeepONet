from gpu_run import run_on_device
from train import train_dense_network, train_don, train_potential
from plot_prediction import plot_random_prediction, plot_prediction_2D
from load_solutions import load_h5_solutions

## Normal Derivative Model
#run_on_device(train_dense_network, "models/fourier_features.keras")
#run_on_device(plot_random_prediction, "models/fourier_features.keras")
#train_don("models/don_model.keras")
#run_on_device(plot_random_prediction, "models/don_model.keras", don=True)


## Potential Model
#run_on_device(train_potential, "models/potential_model_light.keras")
mu, x, y, potential, grad_x, grad_y = load_h5_solutions()
# COUNTOUR PLOT TBD
plot_prediction_2D(mu, x, y, potential, "models/potential_model_light.keras")