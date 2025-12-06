## @package train

import tensorflow as tf
from sklearn.model_selection import train_test_split
from model import DenseNetwork, DeepONet
import numpy as np
from losses import masked_mse, masked_mae


def train(model_path: str, 
          r: int,
          x: np.ndarray,
          y:  np.ndarray,
          mu: np.ndarray,
          seed: int = 40) -> None:

    # Hyperparameters setup ----------------------------------------------------------
    r    = r             # low-rank dimension
    p    = mu.shape[1]   # number of problem parameters = geometrical parameters
    d    = x.shape[2]    # number of spatial dimensions
    ns   = mu.shape[0]   # number of samples = number of meshes
    nh   = x.shape[1]    # max number of dofs = x-coordinates available for each mesh
    seed = seed          # random seed for data splitting
    # --------------------------------------------------------------------------------

    # Print shapes
    print("mu shape:", mu.shape)
    print("x shape:", x.shape)
    print("y shape:", y.shape)

    # Split indices for train+val and test sets first (split along the first dimension)
    idx = np.arange(ns)
    idx_trainval, idx_test = train_test_split(idx, test_size=0.2, random_state=seed)
    # Split train+val indices into train and val sets
    idx_train, idx_val = train_test_split(idx_trainval, test_size=0.2, random_state=seed)
    # Use indices to split the arrays along the first dimension
    mu_train, mu_val, mu_test = mu[idx_train], mu[idx_val], mu[idx_test]
    x_train, x_val, x_test = x[idx_train], x[idx_val], x[idx_test]
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    # Print shapes of the splits
    print("mu_train shape:", mu_train.shape)
    print("X_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    # Mixed Precision Setup
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)


    branch = DenseNetwork(
        normalization_layer=True,
        input_neurons = p, 
        n_neurons = [32], 
        activation = 'relu', 
        output_neurons = r, 
        output_activation = 'linear', 
        initializer = 'he_normal',
        l1_coeff= 0, 
        l2_coeff = 1e-4, 
        batch_normalization = True, 
        dropout = True, 
        dropout_rate = 0.5, 
        leaky_relu_alpha = None,
        layer_normalization = True,
        positional_encoding_frequencies = 0,
    )
    branch.adapt(mu_train)

    trunk = DenseNetwork(
        normalization_layer=True,
        input_neurons = d, 
        n_neurons = [32], 
        activation = 'relu', 
        output_neurons = r, 
        output_activation = 'linear', 
        initializer = 'he_normal',
        l1_coeff= 0, 
        l2_coeff = 1e-4, 
        batch_normalization = True, 
        dropout = True, 
        dropout_rate = 0.5, 
        leaky_relu_alpha = None,
        layer_normalization = True,
        positional_encoding_frequencies = 10,
    )
    trunk.adapt(x_train)

    model = DeepONet(branch = branch, trunk = trunk)

    model.build(input_shape=[(None, p), (None, d)])
    model.summary()

    # --- Learning rate schedule ---
    def lr_warmup_schedule(epoch, lr):
        warmup_epochs = 5
        base_lr = 1e-3
        start_lr = 1e-6
        if epoch <= warmup_epochs:
            return start_lr + (base_lr - start_lr) * (epoch / warmup_epochs)
        return lr

    warmup_callback = tf.keras.callbacks.LearningRateScheduler(lr_warmup_schedule, verbose=0)

    reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        verbose=1
    )

    model.train_model(
        X = x_train,
        mu = mu_train,
        y = y_train,
        X_val = x_val,
        mu_val = mu_val,
        y_val = y_val,
        learning_rate= 1e-3, 
        epochs = 1000, 
        batch_size = 8, 
        loss = masked_mse, 
        validation_freq = 1, 
        verbose = 1, 
        lr_scheduler = [warmup_callback, reduce_callback], 
        metrics = [masked_mae, masked_mse],
        clipnorm = 1, 
        early_stopping_patience = 15,
        log = True,
        optimizer = 'adam')

    print("Evaluating the model on the validation set...")
    model.evaluate([mu_val, x_val], y_val)

    print("Evaluating the model on the test set...")
    model.evaluate([mu_test, x_test], y_test)

    model.save(model_path)

    model.plot_training_history()