## @package losses

import tensorflow as tf

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