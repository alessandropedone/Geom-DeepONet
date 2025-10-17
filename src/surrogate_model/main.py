from gpu_run import run_on_device
from train import train_dense_network, train_don, train_potential
from plot_prediction import plot_random_prediction
#run_on_device(train_dense_network, "models/fourier_features.keras")
#run_on_device(plot_random_prediction, "models/fourier_features.keras")
#train_don("models/don_model.keras")
#run_on_device(plot_random_prediction, "models/don_model.keras", don=True)
train_potential("models/potential_model.keras")