
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from model import DenseNetwork, DeepONet
from meshes_coordinates import load_h5_solutions

def train_dense_network(model_path: str, seed: int = 40):

    # Import coordinates dataset and convert to numpy array
    coordinates = pd.read_csv('data/unrolled_normal_derivative_potential.csv')
    x = coordinates.iloc[:, 1:5]
    x = x.to_numpy()

    # Import normal derivative potential dataset and convert to numpy array
    normal_derivative = pd.read_csv('data/unrolled_normal_derivative_potential.csv')
    y = normal_derivative.iloc[:, 5]
    y = y.to_numpy()    

    seed = seed

    # Split into train+val and test sets first
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        x, y, test_size=0.2, random_state=seed
    )

    # Split train+val into train and val sets
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval, y_trainval, test_size=0.2, random_state=seed
    )

    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_val shape:", x_val.shape)
    print("y_val shape:", y_val.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    # Mixed Precision Setup
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    model = DenseNetwork(
        X = x_train, 
        input_neurons = 4, 
        n_neurons = [512, 256, 128, 256], 
        activation = 'relu', 
        output_neurons = 1, 
        output_activation = 'linear', 
        initializer = 'he_normal',
        l1_coeff= 0, 
        l2_coeff = 1e-4, 
        batch_normalization = True, 
        dropout = True, 
        dropout_rate = 0.5, 
        leaky_relu_alpha = None,
        layer_normalization = True,
        positional_encoding_frequencies = 20,
    )

    # Build the model by providing an input shape just for summary purpose
    # The model will be built in any case during the first call to fit(), which is inside train_model() method
    model.build(input_shape=(None, 4))
    model.summary()

    def lr_warmup_schedule(epoch, lr):
        warmup_epochs = 5
        base_lr = 5e-4
        start_lr = 1e-6
        if epoch <= warmup_epochs:
            return start_lr + (base_lr - start_lr) * (epoch / warmup_epochs)
        return lr
    
    warmup_callback = tf.keras.callbacks.LearningRateScheduler(lr_warmup_schedule, verbose=0)

    reduce_callback = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, verbose=1)

    model.train_model(
        X = x_train, 
        y = y_train, 
        X_val = x_val, 
        y_val = y_val, 
        learning_rate = 1e-3, 
        epochs = 1000, 
        batch_size = 2048, 
        loss = 'mse', 
        validation_freq = 1, 
        verbose = 1, 
        lr_scheduler = [warmup_callback, reduce_callback], 
        metrics = ['mae', 'mse'],
        clipnorm = 1, 
        early_stopping_patience = 15,
        log = True,
        optimizer = 'adam',
    )

    print("Evaluating the model on the validation set...")
    model.evaluate(x = x_val, y = y_val, return_dict=True)

    print("Evaluating the model on the test set...")
    model.evaluate(x = x_test, y = y_test, return_dict=True)

    model.save(model_path)

    model.plot_training_history()
    



def train_don(model_path: str, seed: int = 40):

    import numpy as np

    # Hyperparameters setup --------------------------------------------------------
    r = 20         # low-rank dimension
    p = 3          # number of problem parameters = geometrical parameters
    d = 1          # number of spatial dimensions
    ns = 1000      # number of samples = number of meshes
    nh = 350       # number of dofs = x-coordinates available for each mesh
    seed = seed    # random seed for data splitting
    # ------------------------------------------------------------------------------

    data_csv = pd.read_csv('data/unrolled_normal_derivative_potential.csv')
    mu = data_csv.iloc[:, 1:4] # geometrical parameters
    mu = np.array(mu)

    #Now we have to reshape mu to be ns x p (remove duplicates)
    mu = mu[::nh, :]
    x = data_csv.iloc[:, 4]    # x-coordinates
    x = np.array(x)

    #Now we have to reshape x to be ns x nh x d
    x = x.reshape((ns, nh, d))
    y = data_csv.iloc[:, 5]    # solution at x-coordinates
    y = np.array(y)

    #Now we have to reshape y to be ns x nh
    y = y.reshape((ns, nh))
    print(mu.shape) # should be ns x p
    print(x.shape) # should be ns x nh x d
    print(y.shape) # should be ns x nh

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

    print("mu_train shape:", mu_train.shape)
    print("X_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    # Mixed Precision Setup
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)


    branch = DenseNetwork(
        X = mu_train, 
        input_neurons = p, 
        n_neurons = [512, 256, 128, 256], 
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

    trunk = DenseNetwork(
        X = x_train, 
        input_neurons = d, 
        n_neurons = [512, 256, 128, 256], 
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
        positional_encoding_frequencies = 20,
    )

    model = DeepONet(branch = branch, trunk = trunk)

    model.build(input_shape=[(None, p), (None, d)])
    model.summary()

    # --- Learning rate schedule ---
    def lr_warmup_schedule(epoch, lr):
        warmup_epochs = 5
        base_lr = 5e-4
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
        loss = 'mse', 
        validation_freq = 1, 
        verbose = 1, 
        lr_scheduler = [warmup_callback, reduce_callback], 
        metrics = ['mae', 'mse'],
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


def train_potential(model_path: str, seed: int = 40):

    import numpy as np

    # Import coordinates dataset and solutions
    x1, x2, pot, gx, gy = load_h5_solutions()
    x = np.stack((x1, x2), axis=2)

    # Hyperparameters setup --------------------------------------------------------
    r = 20          # low-rank dimension
    p = 3           # number of problem parameters = geometrical parameters
    d = 2           # number of spatial dimensions
    ns = 1000       # number of samples = number of meshes
    nh = len(x1[0]) # max number of dofs = x-coordinates available for each mesh
    seed = seed     # random seed for data splitting
    # ------------------------------------------------------------------------------

    data_csv = pd.read_csv('data/parameters.csv')
    mu = data_csv.iloc[:, 1:4] # geometrical parameters
    mu = np.array(mu)

    y = pot   # solution at x-coordinates
    y = np.array(y)

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

    print("mu_train shape:", mu_train.shape)
    print("X_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)

    # Mixed Precision Setup
    policy = tf.keras.mixed_precision.Policy('float32')
    tf.keras.mixed_precision.set_global_policy(policy)


    branch = DenseNetwork(
        X = mu_train, 
        input_neurons = p, 
        n_neurons = [128, 64, 32, 128], 
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
        layer_normalization = False,
        positional_encoding_frequencies = 0,
    )

    trunk = DenseNetwork(
        X = x_train, 
        input_neurons = d, 
        n_neurons = [128, 64, 32, 128], 
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
        layer_normalization = False,
        positional_encoding_frequencies = 20,
    )

    model = DeepONet(branch = branch, trunk = trunk)

    model.build(input_shape=[(None, p), (None, d)])
    model.summary()

    # --- Learning rate schedule ---
    def lr_warmup_schedule(epoch, lr):
        warmup_epochs = 5
        base_lr = 5e-4
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

    # Be very careful about precision, since here tf.reduce_sum(mask) 
    # can be very big if the number of points in a batch is large.
    # Here we use float32 globally to avoid issues, 
    # that ensure that we don't get overflow in the summation 
    # when the number of points is less than 3.4e38.
    # Using mixed precision with float16 would lead to overflow
    # when the number of points in batch is larger than ~6e4.
    def masked_mse(y_true, y_pred):
        mask = tf.cast(~tf.math.is_nan(y_true), y_pred.dtype)
        y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_sum(mask * tf.square(y_pred - y_true)) / tf.reduce_sum(mask)

    def masked_mae(y_true, y_pred):
        mask = tf.cast(~tf.math.is_nan(y_true), y_pred.dtype)
        y_true = tf.where(tf.math.is_nan(y_true), 0.0, y_true)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_sum(mask * tf.abs(y_pred - y_true)) / tf.reduce_sum(mask)

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

    

