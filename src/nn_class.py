import tensorflow as tf

class DenseNetwork(tf.keras.Model):
    """ A basic dense network of fixed width """

    def __init__(
        self, 
        width : int, 
        depth : int, 
        output_dim : int,
        input_dim = None,
        activation = 'gelu', 
        kernel_initializer = 'he_uniform'
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
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
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
    



class DeepONet(tf.keras.Model):

    def __init__(
            self, 
            branch : DenseNetwork, 
            trunk : DenseNetwork
        ):
        """
        
        Args:
            branch (DenseNetwork): the branch net to generate expansion coeffs. 
            trunk (DenseNetwork): the trunk net to generate the basis functions.
        """
        super().__init__()
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