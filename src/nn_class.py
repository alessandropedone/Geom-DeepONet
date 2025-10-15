import keras
import tensorflow as tf

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@keras.saving.register_keras_serializable(package="MyLayers")
class DenseNetwork(tf.keras.layers.Layer):
    """ A basic dense network of fixed width """

    def __init__(
        self, 
        width : int = 1, 
        depth : int = 1, 
        output_dim : int = 1,
        input_dim = None,
        activation = 'gelu', 
        kernel_initializer = 'he_uniform',
        **kwargs
    ):
        """
        
        Args:
            width (int): number of neurons in each layer.
            depth (int): number of layers.
            output_dim (int): the output_dimension.
            input_dim (int): the input dimension. Not used for construction, 
                             only for external checking (defaults to 1).
            activation: the activation function (defaults to 'gelu').
            kernel_initializer: weight matrix initializer (defaults to 
                                'he_uniform').
        """
        super(DenseNetwork, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.dense_layers = list()
        for d in range(depth):
            if d < depth - 1:
                self.dense_layers.append(
                    tf.keras.layers.Dense(
                        width,
                        activation = activation,
                        kernel_initializer = kernel_initializer
                    )
                )
            else:
                self.dense_layers.append(
                    tf.keras.layers.Dense(
                        output_dim,
                        activation = 'linear',
                        kernel_initializer = kernel_initializer
                    )
                )
    
    def call(self, x):
        """
        
        Args:
            x: input tensor.

        Returns:
            the NN output.
        """
        for layer in self.dense_layers:
            x = layer(x)
        return x
    
    def get_config(self):
        # Return the config necessary to reconstruct this model
        base_config = super(DenseNetwork, self).get_config()
        return {
            **base_config,
            "width": self.width,
            "depth": self.depth,
            "output_dim": self.output_dim,
            "input_dim": self.input_dim,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer
        }
    



@keras.saving.register_keras_serializable()
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
        assert branch.output_dim == trunk.output_dim
        self.branch = branch
        self.trunk = trunk

    def call(self, inputs):
        """ 

        Args:
            mu: batch of parameters (dimension: B x p)
            x: batch of spatial points (tensor whose last dimension is d)
        """
        mu, x = inputs
        coeffs = self.branch(mu) # dimension: B x r
        basis = self.trunk(x)    # dimension: ... x r
        ein_syntax = 'bj,bij->bi' if (len(basis.shape) == 3) else 'bj,ij->bi'
        return tf.einsum(ein_syntax, coeffs, basis) # dimension: B x ...
    
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