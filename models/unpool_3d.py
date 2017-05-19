import tensorflow as tf
import numpy as np

def unpool_2d_zero_filled(x):
  # https://github.com/tensorflow/tensorflow/issues/2169
  out = tf.concat([x, tf.zeros_like(x)], 3)
  out = tf.concat([out, tf.zeros_like(out)], 2)

  sh = x.get_shape().as_list()
  out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3]]
  return tf.reshape(out, out_size)

def unpool_3d_zero_filled(x):
  # https://github.com/tensorflow/tensorflow/issues/2169
  out = tf.concat([x, tf.zeros_like(x)], 4)
  out = tf.concat([out, tf.zeros_like(out)], 3)
  out = tf.concat([out, tf.zeros_like(out)], 2)

  sh = x.get_shape().as_list()
  out_size = [-1, sh[1] * 2, sh[2] * 2, sh[3] * 2, sh[4]]
  return tf.reshape(out, out_size)


def test_unpoll_2d():
  with tf.Graph().as_default():
    with tf.variable_scope("test_unpoll_2d"):

      _shape = (1, 2, 2, 3)
      x_placeholder = tf.placeholder(tf.float32, shape=_shape)
      output = unpool_2d_zero_filled(x_placeholder)

      init = tf.global_variables_initializer()
      with tf.Session() as session:
        session.run(init)
        x = np.arange(np.prod(_shape)).reshape(_shape)
        _output = session.run(output, feed_dict={x_placeholder: x})
        print(x)
        print(_output)

def test_unpoll_3d():
  with tf.Graph().as_default():
    with tf.variable_scope("test_unpoll_3d"):

      _shape = (1, 2, 2, 2, 3)
      x_placeholder = tf.placeholder(tf.float32, shape=_shape)
      output = unpool_3d_zero_filled(x_placeholder)

      init = tf.global_variables_initializer()
      with tf.Session() as session:
        session.run(init)
        x = np.arange(np.prod(_shape)).reshape(_shape)
        _output = session.run(output, feed_dict={x_placeholder: x})
        print(x)
        print(_output)


if __name__ == "__main__":
  # test_unpoll_2d()
  test_unpoll_3d()
