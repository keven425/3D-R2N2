from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np


class GRU3dCell(tf.contrib.rnn.RNNCell):
    """Wrapper around our GRU cell implementation
    """

    def __init__(self, input_size, state_size):
        """
            input_size: (scalar)
            state_size: (N, N, N, N_h)
        """
        self.input_size = input_size
        self._state_size = state_size

    @property
    def state_size(self):
        # state is actually 4D. but in between ops, we flatten state.
        # because tensorflow doesn't seem to support tuple state_size
        return np.prod(self._state_size)

    @property
    def output_size(self):
        return np.prod(self._state_size)

    # def zero_state(self, batch_size, dtype):
    #     shape = [1] + list(self._state_size)
    #     return tf.zeros(shape, dtype=tf.float32)

    def __call__(self, inputs, state, scope=None):
        """Updates the state using the previous @state and @inputs.

        Args:
            inputs: (None, scalar)
            state: (None, N, N, N, n_h)
            scope: is the name of the scope to be used when defining the variables inside.
        Returns:
            a pair of the output vector and the new state vector.
        """
        scope = scope or type(self).__name__

        with tf.variable_scope(scope):
            shape = [-1] + list(self._state_size)
            _state = tf.reshape(state, shape=shape)
            batch_size = inputs.get_shape()[0].value
            n_h = self._state_size[3]
            W_shape = [self.input_size] + list(self._state_size)

            W_f = tf.get_variable("W_f", shape=W_shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
            b_f = tf.get_variable("b_f", shape=self._state_size, dtype=np.float32)
            W_i = tf.get_variable("W_i", shape=W_shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
            b_i = tf.get_variable("b_i", shape=self._state_size, dtype=np.float32)
            W_s = tf.get_variable("W_s", shape=W_shape, initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
            b_s = tf.get_variable("b_s", shape=self._state_size, dtype=np.float32)

            Uf_h = tf.layers.conv3d(
                _state,
                filters=n_h,
                kernel_size=[3, 3, 3],
                strides=(1, 1, 1),
                padding='same',
                activation=tf.nn.relu,
                use_bias=False,
                name="gru_conv3d_f",
                reuse=False
            )
            z_t = tf.nn.sigmoid(tf.einsum('ij,jklmo->iklmo', inputs, W_i) + Uf_h + b_i)

            Ui_h = tf.layers.conv3d(
                _state,
                filters=n_h,
                kernel_size=[3, 3, 3],
                strides=(1, 1, 1),
                padding='same',
                activation=tf.nn.relu,
                use_bias=False,
                name="gru_conv3d_i",
                reuse=False
            )
            r_t = tf.nn.sigmoid(tf.einsum('ij,jklmo->iklmo', inputs, W_f) + Ui_h + b_f)

            Us_h = tf.layers.conv3d(
                r_t * _state,
                filters=n_h,
                kernel_size=[3, 3, 3],
                strides=(1, 1, 1),
                padding='same',
                activation=tf.nn.relu,
                use_bias=False,
                name="gru_conv3d_s",
                reuse=False
            )
            o_t = tf.nn.tanh(tf.einsum('ij,jklmo->iklmo', inputs, W_s) + Us_h + b_s)

            h_t = z_t * _state + (1 - z_t) * o_t
            h_t = tf.reshape(h_t, shape=(-1, self.state_size))
            new_state = h_t

        # For a GRU, the output and state are the same (N.B. this isn't true
        # for an LSTM, though we aren't using one of those in our
        # assignment)
        output = new_state
        return output, new_state

if __name__ == "__main__":
    pass