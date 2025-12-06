import tensorflow as tf
import pandas as pd
import numpy as np
from model import DenseNetwork, FourierFeatures, LogUniformFreqInitializer, EinsumLayer, DeepONet
from surrogate_model.losses import masked_mse, masked_mae

##
# @param x (numpy.ndarray): The input data for the model.
# @param y (numpy.ndarray): The target data for the model.
# @param model_path (str): The path to the saved model.
def plot_prediction(x: np.ndarray, y: np.ndarray, model_path: str, don : bool = False) -> None:
    """"
    Plot the prediction from the surrogate model.
    """
    # Load the saved model
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={
            "DenseNetwork": DenseNetwork, 
            'FourierFeatures': FourierFeatures, 
            'LogUniformFreqInitializer': LogUniformFreqInitializer, 
            'EinsumLayer': EinsumLayer, 
            'DeepONet': DeepONet})
    
    model.summary()

    # Make a prediction
    if don:
        mu_branch = x[0, 0:3].reshape(1, 3)
        x_trunk = x[:, 3:4]
        y_pred = model([mu_branch, x_trunk]).numpy().flatten()
    else:
        y_pred = model.predict(x)

    # Plot the prediction
    # Take x coordinates from the 4th column
    coords = x[:, 3]
    # Sort the coordinates and corresponding y and y_pred values
    sorted_indices = np.argsort(coords)
    coords = coords[sorted_indices]
    y = y[sorted_indices]
    y_pred = y_pred[sorted_indices]

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.plot(coords, y, label="Normal derivative values", color="blue", linestyle="-")
    plt.plot(coords, y_pred, label=f"Model prediction", color="red", linestyle="--")
    plt.xlim(-50, 50)
    plt.xlabel("coords")
    plt.ylabel("normal_derivative_potential")
    plt.title("Surrogate model prediction test")
    plt.legend()
    plt.grid(True)
    #plt.show()

    # Plot the absolute difference between prediction and true values
    abs_diff = np.abs(y - y_pred)
    mse = np.mean((y - y_pred) ** 2)
    plt.figure(2)
    plt.plot(coords, abs_diff, label="Absolute Error", color="green")
    plt.xlim(-50, 50)
    plt.xlabel("coords")
    plt.ylabel("Absolute Error")
    plt.title(f"Absolute Error of Surrogate Model Prediction")
    mean_abs_error = np.mean(abs_diff)
    plt.axhline(mean_abs_error, color="orange", linestyle="--", label=f"Mean Abs Error: {mean_abs_error:.4f}")
    plt.axhline(np.sqrt(mse), color="purple", linestyle="--", label=f"RMSE: {np.sqrt(mse):.4f}")
    plt.axhline(mse, color="red", linestyle="--", label=f"MSE: {mse:.6f}")
    plt.legend()
    plt.grid(True)
    plt.show()

##
# @param model_path (str): The path to the saved model.
def plot_random_prediction(model_path: str, don: bool = False) -> None:
    """
    Plot a random prediction from the surrogate model.
    """
    # Import parameters, coordinates and normal derivative dataset
    data = pd.read_csv('data/unrolled_normal_derivative_potential.csv')

    # Save parameters and coordinates and convert to numpy array
    x = data.iloc[:, 1:5]
    x = x.to_numpy()

    # Save normal derivative potential values and convert to numpy array
    y = data.iloc[:, 5]
    y = y.to_numpy()

    # Predict on a random test element
    # Select a random test element
    import random

    # Select random combination of the 3 geometric parameters
    random_index = random.randint(0, len(x) - 1)
    triplet = x[random_index, 0:3]

    # Now retrieve all the coordinates that match this triplet
    mask = np.all(x[:, 0:3] == triplet, axis=1)

    x_sample = x[mask]
    y_sample = y[mask]

    plot_prediction(x_sample, y_sample, model_path=model_path, don=don)



import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

def plot_comparison(x, y, z_real, z_approx, title="Function Comparison"):
    """
    Plot comparison between real function and approximation in 3D
    
    Parameters:
    x, y: 1D arrays of point coordinates
    z_real: real function values at points
    z_approx: approximated function values at points
    title: plot title
    """
    fig = plt.figure(figsize=(25, 10))
    
    # Create grid for contour plots
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    Xi, Yi = np.meshgrid(xi, yi)
    
    # Interpolate scattered data to grid
    Zi_real = griddata((x, y), z_real, (Xi, Yi), method='linear')
    Zi_approx = griddata((x, y), z_approx, (Xi, Yi), method='linear')
    
    # Real function
    ax1 = fig.add_subplot(1, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(Xi, Yi, Zi_real, cmap='viridis', alpha=0.8)
    # draw vertical red lines from z_real to z_approx at each (x,y)
    for xi, yi, zr, za in zip(x, y, z_real, z_approx):
        ax1.plot([xi, xi], [yi, yi], [zr, za], color='red', linewidth=0.7, alpha=0.8)
    ax1.set_title('Real Function')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)

    ax7 = fig.add_subplot(1, 3, 2, projection='3d')
    # draw vertical red lines from z_real to z_approx at each (x,y)
    for xi, yi, zr, za in zip(x, y, z_real, z_approx):
        ax7.plot([xi, xi], [yi, yi], [zr, za], color='red', linewidth=0.7, alpha=0.8)
    ax7.set_title('Error Lines')
    ax7.set_xlabel('x')
    ax7.set_ylabel('y')
    ax7.set_zlabel('z')
    fig.colorbar(surf1, ax=ax7, shrink=0.5)
    
    # Approximation
    ax2 = fig.add_subplot(1, 3, 3, projection='3d')
    surf2 = ax2.plot_surface(Xi, Yi, Zi_approx, cmap='viridis', alpha=0.8)
    #ax2.scatter(x, y, z_approx, c='red', s=10)  # Show original points
    ax2.set_title('Approximation')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    fig.colorbar(surf2, ax=ax2, shrink=0.5)
    
    fig2 = plt.figure(figsize=(25, 10))
    # Error/Difference + histogram inset
    ax3 = fig2.add_subplot(1, 3, 1)
    error = np.abs(z_real - z_approx)
    mse = np.nanmean(error**2)
    scatter = ax3.scatter(x, y, c=error, cmap='coolwarm', s=2)
    ax3.set_title(f'Mean Square Error (MSE: {mse:.4f})')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    # remove hard-coded limits to keep generality
    ax3.set_ylim(-5, 5)
    ax3.set_xlim(-55, -45)
    fig2.colorbar(scatter, ax=ax3, shrink=0.5)

    ax4 = fig2.add_subplot(1, 3, 2)
    error = np.abs(z_real - z_approx)
    mse = np.nanmean(error**2)
    scatter = ax4.scatter(x, y, c=error, cmap='coolwarm', s=2)
    ax4.set_title(f'Mean Square Error (MSE: {mse:.4f})')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    # remove hard-coded limits to keep generality
    ax4.set_ylim(-8, 8)
    ax4.set_xlim(-30, 30)
    fig2.colorbar(scatter, ax=ax4, shrink=0.5)

    ax5 = fig2.add_subplot(1, 3, 3)
    error = np.abs(z_real - z_approx)
    rmse = np.sqrt(np.nanmean(error**2))
    scatter = ax5.scatter(x, y, c=error, cmap='coolwarm', s=2)
    ax5.set_title(f'Root Mean Square Error (RMSE: {rmse:.4f})')
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    # remove hard-coded limits to keep generality
    ax5.set_ylim(-5, 5)
    ax5.set_xlim(45, 55)
    fig2.colorbar(scatter, ax=ax5, shrink=0.5)

    # New separate figure: error histogram
    err_flat = error.flatten()
    err_flat = err_flat[~np.isnan(err_flat)]
    plt.figure(figsize=(10, 10))
    plt.hist(err_flat, bins=30, color='gray', alpha=0.8)
    plt.axvline(np.mean(err_flat), color='red', linestyle='--', linewidth=1,
                label=f"Mean: {np.mean(err_flat):.4f}")
    plt.title('Error Histogram')
    plt.xlabel('Absolute error')
    plt.ylabel('Counts')
    plt.legend()
    plt.tight_layout()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_prediction_2D(mu: np.array, x: np.ndarray, y: np.ndarray, z: np.ndarray, model_path: str) -> None:
    x = x[0].reshape(1, x.shape[1])
    y = y[0].reshape(1, y.shape[1])
    coord = np.stack((x, y), axis=2)
    mu = mu[0].reshape(1, mu.shape[1])
    z_real = z[0].reshape(1, z.shape[1])

    # Load the saved model
    model = tf.keras.models.load_model(
        model_path, 
        custom_objects={
            "DenseNetwork": DenseNetwork, 
            'FourierFeatures': FourierFeatures, 
            'LogUniformFreqInitializer': LogUniformFreqInitializer, 
            'EinsumLayer': EinsumLayer, 
            'DeepONet': DeepONet,
            'masked_mse': masked_mse,
            'masked_mae': masked_mae})

    model.summary()

    print ("Shape of mu", mu.shape)
    print ("Shape of coord", coord.shape)
    print ("Shape of z_real", z_real.shape)

    z_pred = model([mu,coord]).numpy()
    branch_output = model.call([mu, coord], return_branch=True).numpy()
    trunk_output = model.call([mu, coord], return_trunk=True).numpy()
    print ("Shape of branch output", branch_output.shape)
    print ("Shape of trunk output", trunk_output.shape)
    print ("Shape of z_pred", z_pred.shape)
    x = x.flatten()
    y = y.flatten()
    z_pred = z_pred.flatten()
    z_real = z_real.flatten()

    #plot_comparison(x, y, z_real, z_pred, title="Potential Function Comparison")





