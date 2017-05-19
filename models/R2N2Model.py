import tensorflow as tf

from models.Model import Model
from models.GRU3dCell import GRU3dCell
import models.unpool_3d

class R2N2Model(Model):

  def __init__(self, config):
    self.config = config
    self.grad_norm = None

    # Defining placeholders.
    self.input_placeholder = None
    self.labels_placeholder = None
    self.dropout_keep_placeholder = None
    self.learning_rate_placeholder = None

    self.build()

  def train_on_batch(self, sess, input_batch, labels_batch, lr):
    """Perform one step of gradient descent on the provided batch of data.

    Args:
        sess: tf.Session()
        input_batch: np.ndarray of shape (None, n_timesteps, height, width, channel)
        labels_batch: np.ndarray of shape (None, x_size, y_size, z_size, 1)
    """
    feed = self.create_feed_dict(input_batch=input_batch,
                                 labels_batch=labels_batch,
                                 lr=lr,
                                 dropout_keep=self.config.dropout_keep)
    _, loss, grad_norm, learning_rate, logits_norm, grads_vars = sess.run(
      [self.train_op, self.loss, self.grad_norm, self.learning_rate_placeholder, self.logits_norm, self.grads_vars], feed_dict=feed)
    return loss, grad_norm, learning_rate, logits_norm, grads_vars

  def evaluate_on_batch(self, sess, input_batch, labels_batch):
    feed = self.create_feed_dict(input_batch=input_batch, labels_batch=labels_batch)
    pred, loss = sess.run([self.pred, self.loss], feed_dict=feed)  # pick the class that has highest probability
    return pred, loss

  def add_placeholders(self):
    self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.max_timestep, self.config.CONST.IMG_H, self.config.CONST.IMG_W, 3))
    self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.CONST.N_VOX, self.config.CONST.N_VOX, self.config.CONST.N_VOX))
    self.dropout_keep_placeholder = tf.placeholder(tf.float32)
    self.learning_rate_placeholder = tf.placeholder(tf.float32)

  def create_feed_dict(self, input_batch, labels_batch=None, lr=None, dropout_keep=1.0):
    feed_dict = {self.input_placeholder: input_batch,
                 self.dropout_keep_placeholder: dropout_keep}

    if not lr is None:
      feed_dict[self.learning_rate_placeholder] = lr
    if not labels_batch is None:
      feed_dict[self.labels_placeholder] = labels_batch

    return feed_dict

  def add_logit_op(self):
    # self.input_placeholder: (None, n_timesteps, height, width, n_channels)

    with tf.variable_scope("R2N2_logit"):

      # reshape to merge n_batches, max_timesteps
      # because conv2d() function only takes 4D tensor, for 2D convolution
      self.input_placeholder.get_shape()
      input = tf.reshape(self.input_placeholder, shape=(-1, self.config.CONST.IMG_H, self.config.CONST.IMG_W, 3))

      # 1st conv layer
      conv11 = tf.contrib.layers.conv2d(inputs=input, num_outputs=64, kernel_size=[7, 7], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv11", reuse=False)
      conv12 = tf.contrib.layers.conv2d(inputs=conv11, num_outputs=64, kernel_size=[3, 3], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv12", reuse=False)
      conv1_res =  tf.contrib.layers.conv2d(inputs=input, num_outputs=64, kernel_size=[1, 1], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv1_res", reuse=False)
      conv1 = conv12 + conv1_res
      conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='conv1_pool')


      # 2nd conv layer
      conv21 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=128, kernel_size=[3, 3], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv21", reuse=False)
      conv22 = tf.contrib.layers.conv2d(inputs=conv21, num_outputs=128, kernel_size=[3, 3], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv22", reuse=False)
      conv2_res = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=128, kernel_size=[1, 1], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv2_res", reuse=False)
      conv2 = conv22 + conv2_res
      conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='conv2_pool')


      # 3rd conv layer
      conv31 = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=256, kernel_size=[3, 3], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv31", reuse=False)
      conv32 = tf.contrib.layers.conv2d(inputs=conv31, num_outputs=256, kernel_size=[3, 3], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv32", reuse=False)
      conv3_res = tf.contrib.layers.conv2d(inputs=conv2, num_outputs=256, kernel_size=[1, 1], stride=1, padding="same",
        activation_fn=tf.nn.relu, scope="conv3_res", reuse=False)
      conv3 = conv32 + conv3_res
      conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='conv3_pool')

      # 4th conv layer
      conv41 = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=256, kernel_size=[3, 3], stride=1, padding="same",
                                        activation_fn=tf.nn.relu, scope="conv41", reuse=False)
      conv42 = tf.contrib.layers.conv2d(inputs=conv41, num_outputs=256, kernel_size=[3, 3], stride=1, padding="same",
                                        activation_fn=tf.nn.relu, scope="conv42", reuse=False)
      conv4_res = tf.contrib.layers.conv2d(inputs=conv3, num_outputs=256, kernel_size=[1, 1], stride=1, padding="same",
                                           activation_fn=tf.nn.relu, scope="conv4_res", reuse=False)
      conv4 = conv42 + conv4_res
      conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='conv4_pool')

      # 5th conv layer
      conv51 = tf.contrib.layers.conv2d(inputs=conv4, num_outputs=256, kernel_size=[3, 3], stride=1, padding="same",
                                        activation_fn=tf.nn.relu, scope="conv51", reuse=False)
      conv52 = tf.contrib.layers.conv2d(inputs=conv51, num_outputs=256, kernel_size=[3, 3], stride=1, padding="same",
                                        activation_fn=tf.nn.relu, scope="conv52", reuse=False)
      conv5_res = tf.contrib.layers.conv2d(inputs=conv4, num_outputs=256, kernel_size=[1, 1], stride=1, padding="same",
                                           activation_fn=tf.nn.relu, scope="conv5_res", reuse=False)
      conv5 = conv52 + conv5_res
      conv5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='conv5_pool')

      conv5_flattened = tf.contrib.layers.flatten(conv5)

      n_fc_outputs = 1024
      fc = tf.contrib.layers.fully_connected(inputs=conv5_flattened, num_outputs=n_fc_outputs,
        activation_fn=tf.nn.relu, scope="fc1", reuse=False)

      # reshape back to (batch_size, max_timesteps, n_fc_outputs)
      fc = tf.reshape(fc, shape=(-1, self.config.max_timestep, n_fc_outputs))

      # 3D GRU
      grid_state_size = (4, 4, 4, 128)
      cell = GRU3dCell(fc.shape[-1].value, grid_state_size)
      _, h = tf.nn.dynamic_rnn(cell, fc, dtype=tf.float32)
      shape = [-1] + list(grid_state_size)
      h = tf.reshape(h, shape=(shape)) # reshape back to 3d

      # deconvolutional layers
      # 1st deconv layer
      deconv11 = tf.layers.conv3d(h, filters=128, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv11", reuse=False)
      deconv12 = tf.layers.conv3d(deconv11, filters=128, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv12", reuse=False)
      deconv1_res = tf.layers.conv3d(h, filters=128, kernel_size=[1, 1, 1], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv1_res", reuse=False)
      deconv1 = deconv12 + deconv1_res
      deconv1 = models.unpool_3d.unpool_3d_zero_filled(deconv1)


      # 2nd deconv layer
      deconv21 = tf.layers.conv3d(deconv1, filters=64, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv21", reuse=False)
      deconv22 = tf.layers.conv3d(deconv21, filters=64, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv22", reuse=False)
      deconv2_res = tf.layers.conv3d(
        deconv1, filters=64, kernel_size=[1, 1, 1], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv2_res", reuse=False)
      deconv2 = deconv22 + deconv2_res
      deconv2 = models.unpool_3d.unpool_3d_zero_filled(deconv2)


      # 3rd deconv layer
      deconv31 = tf.layers.conv3d(
        deconv2, filters=32, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv31", reuse=False)
      deconv32 = tf.layers.conv3d(
        deconv31, filters=32, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv32", reuse=False)
      deconv3_res = tf.layers.conv3d(
        deconv2, filters=32, kernel_size=[1, 1, 1], strides=(1, 1, 1), padding='same',
        activation=tf.nn.relu, use_bias=False, name="deconv3_res", reuse=False)
      deconv3 = deconv32 + deconv3_res
      deconv3 = models.unpool_3d.unpool_3d_zero_filled(deconv3)


      # final deconv layer
      deconv4 = tf.layers.conv3d(
        deconv3, filters=2, kernel_size=[1, 1, 1], strides=(1, 1, 1), padding='same',
        activation=None, use_bias=False, name="deconv4", reuse=False)

    return fc, h, deconv4

  def add_prediction_op(self, logits):
    fc, h, deconv4 = logits
    return deconv4

  def add_loss_op(self, logits):
    fc, h, deconv4 = logits

    self.logits_norm = tf.sqrt(tf.reduce_mean(tf.square(deconv4)))
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=deconv4, labels=self.labels_placeholder)
    loss = tf.reduce_mean(cross_entropy)
    return loss

  def add_training_op(self, loss):
    batch = tf.Variable(0, trainable=False)
    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
    grads_vars = optimizer.compute_gradients(loss)
    filtered_grads_vars = []
    for grad, var in grads_vars:
      if not grad is None:
        filtered_grads_vars.append((grad, var))
    grads = [pair[0] for pair in filtered_grads_vars]
    self.grad_norm = tf.global_norm(grads)
    self.grads_vars = filtered_grads_vars
    train_op = optimizer.apply_gradients(filtered_grads_vars, global_step=batch)
    return train_op
