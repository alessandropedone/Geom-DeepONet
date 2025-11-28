import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import datetime
from keras.saving import register_keras_serializable
    
@register_keras_serializable()
class FourierFeatures(tf.keras.layers.Layer):
    def __init__(self, num_frequencies, learnable=True, initializer='glorot_uniform', **kwargs):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.learnable = learnable
        self.initializer = initializer

    def build(self, input_shape):
        shape = (1, self.num_frequencies)
        if self.learnable:
            self.freqs = self.add_weight(name="freqs", shape=shape,
                                         initializer=self.initializer,
                                         trainable=True)
        else:
            self.freqs = tf.constant(2.0 ** tf.range(1, self.num_frequencies + 1, dtype=tf.float32)[tf.newaxis, :])

    def call(self, x):
        # Start encoding list
        encoded = []
        # Apply sinusoidal or learnable frequency encoding
        for i in range(self.num_frequencies):
            freq = self.freqs[0, i]
            for j in range(x.shape[-1]): # x shape: (batch, time, feature)
                current_x = x[..., j:j+1]  # shape: (batch, time, 1)
                encoded.append(tf.sin(freq * current_x))
                encoded.append(tf.cos(freq * current_x))
        # Concatenate along the feature dimension
        encoded_features = tf.concat(encoded, axis=-1)
        # Combine the original x with the new encodings
        return tf.concat([x, encoded_features], axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_frequencies": self.num_frequencies,
            "learnable": self.learnable,
            # For initializer, save its config dict if possible
            "initializer": tf.keras.initializers.serialize(self.initializer),
        })
        return config
    

@register_keras_serializable()
class LogUniformFreqInitializer(tf.keras.initializers.Initializer):
    def __init__(self, min_exp=0.0, max_exp=8.0):
        self.min_exp = min_exp
        self.max_exp = max_exp
        
    def __call__(self, shape, dtype=None):
        # Sample uniformly from [min_exp, max_exp]
        exponents = tf.random.uniform(shape, self.min_exp, self.max_exp, dtype=dtype)
        return tf.math.pow(2.0, exponents)

    def get_config(self):
        return {'min_exp': self.min_exp, 'max_exp': self.max_exp}


@register_keras_serializable()
class DenseNetwork(tf.keras.Model):
    """
    A class to build, train, and manage a neural network model using TensorFlow and Keras.
    """

    ##
    def __init__(self,
                normalization_layer: bool = True,
                input_neurons: int = 1, 
                n_neurons: list = None, 
                activation: str = 'tanh',
                output_neurons: int = 1,
                output_activation: str = 'linear',
                initializer: str = 'glorot_uniform',
                l1_coeff: float = 0,
                l2_coeff: float = 0,
                batch_normalization: bool = False,
                dropout: bool = False,
                dropout_rate: float = 0.3,
                leaky_relu_alpha: float = None,
                layer_normalization: bool = False,
                positional_encoding_frequencies: int = 0,
                **kwargs):
        """
        Initializes the NN_Model class with an empty Sequential model and a None history.
        """

        super().__init__(**kwargs)

        self.normalization_layer = normalization_layer
        self.norm_layer = tf.keras.layers.Normalization(axis=-1)
        self.input_neurons = input_neurons
        self.n_neurons = n_neurons or [64] * 8  # safe default
        self.activation = activation
        self.output_neurons = output_neurons
        self.output_activation = output_activation
        self.initializer = initializer
        self.l1_coeff = l1_coeff
        self.l2_coeff = l2_coeff
        self.batch_normalization = batch_normalization
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.leaky_relu_alpha = leaky_relu_alpha
        self.layer_normalization = layer_normalization
        self.positional_encoding_frequencies = positional_encoding_frequencies
        self.all_layers = list()

        # Initialize history to None
        self.history = None


    def build(self, input_shape):
        """
        Builds the model layers based on the input shape.
        Called automatically when the model is first used.
        """
        l1_l2 = tf.keras.regularizers.l1_l2
        Dense = tf.keras.layers.Dense
        BatchNormalization = tf.keras.layers.BatchNormalization
        Dropout = tf.keras.layers.Dropout
        LeakyReLU = tf.keras.layers.LeakyReLU

        self.all_layers = []

        # Normalization
        if self.normalization_layer:
            self.all_layers.append(self.norm_layer)

        # Positional Encoding
        if self.positional_encoding_frequencies and self.positional_encoding_frequencies > 0:
            self.all_layers.append(FourierFeatures(
                num_frequencies=self.positional_encoding_frequencies, 
                learnable=True, 
                initializer=LogUniformFreqInitializer(min_exp=0.0, max_exp=8.0)
            ))

        # First hidden layer
        if self.leaky_relu_alpha is not None:
            self.all_layers.append(Dense(
                self.n_neurons[0], 
                kernel_initializer=self.initializer,
                kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
            ))
            self.all_layers.append(LeakyReLU(alpha=self.leaky_relu_alpha))
        else:
            self.all_layers.append(Dense(
                self.n_neurons[0], 
                activation=self.activation,
                kernel_initializer=self.initializer,
                kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
            ))
        
        if self.batch_normalization:
            self.all_layers.append(BatchNormalization())
        if self.dropout:
            self.all_layers.append(Dropout(self.dropout_rate))

        # Hidden layers
        for neurons in self.n_neurons[1:]:
            if self.leaky_relu_alpha is not None:
                self.all_layers.append(Dense(
                    neurons, 
                    kernel_initializer=self.initializer,
                    kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
                ))
                self.all_layers.append(LeakyReLU(alpha=self.leaky_relu_alpha))
            else:
                self.all_layers.append(Dense(
                    neurons, 
                    activation=self.activation,
                    kernel_initializer=self.initializer,
                    kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
                ))

            if self.batch_normalization:
                self.all_layers.append(BatchNormalization())
            if self.dropout:
                self.all_layers.append(Dropout(self.dropout_rate))
            if self.layer_normalization:
                self.all_layers.append(tf.keras.layers.LayerNormalization())
        
        # Output layer
        self.all_layers.append(Dense(
            self.output_neurons, 
            activation=self.output_activation,
            kernel_regularizer=l1_l2(l1=self.l1_coeff, l2=self.l2_coeff)
        ))

        dummy_input = tf.keras.Input(shape=(self.input_neurons,))
        self.call(dummy_input)

        super(DenseNetwork, self).build(input_shape)

    def call(self, x):
        for layer in self.all_layers:
            x = layer(x)
        return x
    
    def adapt(self, X: np.ndarray):
        if self.normalization_layer:
            if X.ndim == 3:
                self.norm_layer.adapt(X.reshape(-1, X.shape[-1]))
            else:
                self.norm_layer.adapt(X)
    
    def get_config(self):
        # Return the config necessary to reconstruct this model
        base_config = super(DenseNetwork, self).get_config()
        return {
            **base_config,
            "normalization_layer": self.normalization_layer,
            "input_neurons": self.input_neurons,
            "n_neurons": self.n_neurons,
            "activation": self.activation,
            "output_neurons": self.output_neurons,
            "output_activation": self.output_activation,
            "initializer": self.initializer,
            "l1_coeff": self.l1_coeff,
            "l2_coeff": self.l2_coeff,
            "batch_normalization": self.batch_normalization,
            "dropout": self.dropout,
            "dropout_rate": self.dropout_rate,
            "leaky_relu_alpha": self.leaky_relu_alpha,
            "layer_normalization": self.layer_normalization,
            "positional_encoding_frequencies": self.positional_encoding_frequencies,
        }
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
        
    ##
    # @param X (np.ndarray): The input data for training.
    # @param y (np.ndarray): The target data for training.
    # @param X_val (np.ndarray): The input data for validation.
    # @param y_val (np.ndarray): The target data for validation.
    # @param learning_rate (float): The learning rate for the optimizer.
    # @param epochs (int): The number of epochs for training.
    # @param batch_size (int): The size of the batches for training.
    # @param loss (str): The loss function to be used during training.
    # @param validation_freq (int): The frequency of validation during training.
    # @param lr_schedule (Optional[Callable[[int], float]]): A function to adjust the learning rate.
    # @param optimizer (str): The optimizer to be used for training. Options are 'adam', 'sgd', 'rmsprop'.
    # @return None
    # @throws ValueError: If any of the input arrays are empty.
    def train_model(self, 
                    X: np.ndarray, 
                    y: np.ndarray, 
                    X_val: np.ndarray, 
                    y_val: np.ndarray, 
                    learning_rate: float = 1e-3, 
                    epochs: int = 10000, 
                    batch_size: int = 15000, 
                    loss: str = 'mean_squared_error', 
                    validation_freq: int = 1, 
                    verbose: int = 0,
                    lr_scheduler = None,
                    metrics: list = ['mse'],
                    clipnorm: float = None,
                    early_stopping_patience: int = None,
                    log: bool = False,
                    optimizer: str = 'adam'
                    ) -> None:
        """
        Trains the model on the provided dataset.
        Use "tensorboard --logdir logs" to visualize logs (if log is set to True).
        """
        if X.size == 0 or y.size == 0 or X_val.size == 0 or y_val.size == 0:
            raise ValueError("Input arrays must not be empty")
        if loss == 'huber_loss':
            loss = tf.keras.losses.Huber(delta=1.0)

        if optimizer not in ['adam', 'sgd', 'rmsprop']:
            raise ValueError("Unsupported optimizer. Supported optimizers are: 'adam', 'sgd', 'rmsprop'.")
        if optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD
            if clipnorm is not None:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True, clipnorm=clipnorm))
            else:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True))
        elif optimizer == 'rmsprop':
            self.compile(loss=loss, metrics=metrics,optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False))
        elif optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam
            if clipnorm is not None:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, clipnorm=clipnorm))
            else:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate))
        
        callbacks = []
        if lr_scheduler is not None:
            for callback in lr_scheduler:
                callbacks.append(callback)

        # Set up TensorBoard callback with profiling
        if log:
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_steps_per_second=True)
            callbacks.append(tensorboard_callback)
        # TensorBoard command: tensorboard --logdir logs

        # Early stopping callback
        if early_stopping_patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience, 
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        self.history = self.fit(
            X, y, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data=(X_val, y_val), validation_freq=validation_freq,
            callbacks=callbacks
        )

    ##
    def plot_training_history(self) -> None:
        """
        Plots the training and validation loss over epochs.
        This method should be called after training the model using `train_model`.
        """
        if self.history is None:
            raise ValueError("The model has no training history. Train the model using 'train_model' method first.")

        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        tot_train = len(self.history.history['loss'])
        tot_valid = len(self.history.history['val_loss']) 
        valid_freq = int(tot_train / tot_valid)
        plt.plot(np.arange(tot_train), self.history.history['loss'], 'b-', label='Training loss', linewidth=2)
        plt.plot(valid_freq * np.arange(tot_valid), self.history.history['val_loss'], 'r--', label='Validation loss', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.grid(True)
        plt.show()

@register_keras_serializable()
class EinsumLayer(tf.keras.layers.Layer):
    def __init__(self, ein_syntax, **kwargs):
        super().__init__(**kwargs)
        self.ein_syntax = ein_syntax

    def call(self, inputs):
        coeffs, basis = inputs
        return tf.einsum(self.ein_syntax, coeffs, basis)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "ein_syntax": self.ein_syntax,
        })
        return config
    
@register_keras_serializable()
class DeepONet(tf.keras.Model):

    def __init__(
            self, 
            branch : DenseNetwork, 
            trunk : DenseNetwork,
            **kwargs
        ):
        """
        
        Args:
            branch (DenseNetwork): the branch net to generate expansion coeffs. 
            trunk (DenseNetwork): the trunk net to generate the basis functions.
        """
        super(DeepONet, self).__init__(**kwargs)
        assert branch.output_neurons == trunk.output_neurons
        self.branch = branch
        self.trunk = trunk

        # Initialize history to None
        self.history = None

    def call(self, inputs, return_branch=False, return_trunk=False):
        """ 

        Args:
            mu: batch of parameters (dimension: B x p)
            x: batch of spatial points (tensor whose last dimension is d)
        """
        mu, x = inputs

        # Compute branch output
        # Input mu shape: batch x p
        # Output shape: batch x low rank dimension
        coeffs = self.branch(mu)
        
        # Compute trunk output
        # Input x shape: time x d  OR  batch x time x d
        # Output shape: time x low rank dimension  OR  batch x time x low rank dimension
        if len(x.shape) == 3:  # batch x time x d
            # Apply trunk to each time step
            original_shape = tf.shape(x)
            x_reshaped = tf.reshape(x, [-1, x.shape[-1]])
            basis_flat = self.trunk(x_reshaped)
            basis = tf.reshape(basis_flat, [original_shape[0], original_shape[1], -1])
        else:  # time x d
            basis = self.trunk(x)

        if return_branch & return_trunk:
            return coeffs, basis
        if return_branch:
            return coeffs
        if return_trunk:
            return basis
        
        ein_syntax = 'bj,bij->bi' if (len(basis.shape) == 3) else 'bj,ij->bi'
        output = EinsumLayer(ein_syntax)([coeffs, basis])
        return output

    def build(self, input_shape):
        dummy_mu = tf.keras.Input(shape=(self.branch.input_neurons,))
        dummy_x = tf.keras.Input(shape=(self.trunk.input_neurons,))
        self.call((dummy_mu, dummy_x))
        super(DeepONet, self).build(input_shape)
    
    def get_config(self):
        # Return the config necessary to reconstruct this model
        base_config = super(DeepONet, self).get_config()
        return {
            **base_config,
            "branch": tf.keras.utils.serialize_keras_object(self.branch),
            "trunk": tf.keras.utils.serialize_keras_object(self.trunk)
        }

    @classmethod
    def from_config(cls, config):
        branch_config = config.pop("branch")
        trunk_config = config.pop("trunk")
        config["branch"] = tf.keras.utils.deserialize_keras_object(branch_config)
        config["trunk"] = tf.keras.utils.deserialize_keras_object(trunk_config)
        return cls(**config)
    

    def train_model(self, 
                    X: np.ndarray,
                    mu: np.ndarray,
                    y: np.ndarray, 
                    X_val: np.ndarray, 
                    mu_val: np.ndarray,
                    y_val: np.ndarray, 
                    learning_rate: float = 1e-3, 
                    epochs: int = 10000, 
                    batch_size: int = 15000, 
                    loss: str = 'mean_squared_error', 
                    validation_freq: int = 1, 
                    verbose: int = 0,
                    lr_scheduler = None,
                    metrics: list = ['mse'],
                    clipnorm: float = None,
                    early_stopping_patience: int = None,
                    log: bool = False,
                    optimizer: str = 'adam'
                    ) -> None:
        """
        Trains the model on the provided dataset.
        Use "tensorboard --logdir logs" to visualize logs (if log is set to True).
        """
        if X.size == 0 or y.size == 0 or X_val.size == 0 or y_val.size == 0 or mu.size == 0 or mu_val.size == 0:
            raise ValueError("Input arrays must not be empty")
        if loss == 'huber_loss':
            loss = tf.keras.losses.Huber(delta=1.0)

        if optimizer not in ['adam', 'sgd', 'rmsprop']:
            raise ValueError("Unsupported optimizer. Supported optimizers are: 'adam', 'sgd', 'rmsprop'.")
        if optimizer == 'sgd':
            optimizer = tf.keras.optimizers.SGD
            if clipnorm is not None:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True, clipnorm=clipnorm))
            else:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, momentum=0.9, nesterov=True))
        elif optimizer == 'rmsprop':
            self.compile(loss=loss, metrics=metrics,optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate, rho=0.9, momentum=0.9, epsilon=1e-07, centered=False))
        elif optimizer == 'adam':
            optimizer = tf.keras.optimizers.Adam
            if clipnorm is not None:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate, clipnorm=clipnorm))
            else:
                self.compile(loss=loss, metrics=metrics,optimizer=optimizer(learning_rate=learning_rate))
        
        callbacks = []
        if lr_scheduler is not None:
            for callback in lr_scheduler:
                callbacks.append(callback)

        # Set up TensorBoard callback with profiling
        if log:
            log_dir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_steps_per_second=True)
            callbacks.append(tensorboard_callback)
        # TensorBoard command: tensorboard --logdir logs

        # Early stopping callback
        if early_stopping_patience is not None:
            early_stopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=early_stopping_patience, 
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        self.history = self.fit(
            [mu, X], y, epochs=epochs, batch_size=batch_size, verbose=verbose,
            validation_data=([mu_val, X_val], y_val), validation_freq=validation_freq,
            callbacks=callbacks
        )

    ##
    def plot_training_history(self) -> None:
        """
        Plots the training and validation loss over epochs.
        This method should be called after training the model using `train_model`.
        """
        if self.history is None:
            raise ValueError("The model has no training history. Train the model using 'train_model' method first.")

        plt.figure(figsize=(8, 6))  # Adjust the figure size as needed
        tot_train = len(self.history.history['loss'])
        tot_valid = len(self.history.history['val_loss']) 
        valid_freq = int(tot_train / tot_valid)
        plt.plot(np.arange(tot_train), self.history.history['loss'], 'b-', label='Training loss', linewidth=2)
        plt.plot(valid_freq * np.arange(tot_valid), self.history.history['val_loss'], 'r--', label='Validation loss', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epochs', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend(fontsize=12)
        plt.title('Training and Validation Loss', fontsize=16)
        plt.grid(True)
        plt.show()