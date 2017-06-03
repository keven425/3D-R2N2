import tensorflow as tf
import numpy as np
from scipy import misc
from PIL import Image


def rotate_image(images, angles):
  return tf.contrib.image.rotate(images, angles)

def load_image(infilename):
  img = misc.imread(infilename)
  return img


def save_image(npdata, outfilename):
  misc.imsave(outfilename, npdata)

def test_rotate():
  with tf.Graph().as_default():
    with tf.variable_scope("test_image_rotate"):

      _shape = (1, 500, 500, 3)
      x_placeholder = tf.placeholder(tf.float32, shape=_shape)
      angles_placeholder = tf.placeholder(tf.float32, shape=(1,))
      output = rotate_image(x_placeholder, angles_placeholder)

      init = tf.global_variables_initializer()
      with tf.Session() as session:
        session.run(init)
        x = load_image('./test_rotate/img.jpg')
        x = x.reshape(_shape)
        angles = [0.1]
        _output = session.run(output, feed_dict={
          x_placeholder: x,
          angles_placeholder: angles})
        _output = _output.reshape((500, 500, 3))
        save_image(_output, './test_rotate/img_rotated.jpg')


if __name__ == "__main__":
  # test_unpoll_2d()
  test_rotate()
