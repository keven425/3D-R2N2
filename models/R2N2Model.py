import tensorflow as tf
import numpy as np

from models.Model import Model
from models.GRU3dCell import GRU3dCell
import models.unpool_3d

import lib.voxel

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

  def train_on_batch(self, sess, input_batch, labels_batch, poses_batch, lr):
    """Perform one step of gradient descent on the provided batch of data.

    Args:
        sess: tf.Session()
        input_batch: np.ndarray of shape (None, n_timesteps, height, width, channel)
        labels_batch: np.ndarray of shape (None, x_size, y_size, z_size, 1)
    """
    feed = self.create_feed_dict(is_training=True,
                                 input_batch=input_batch,
                                 labels_batch=labels_batch,
                                 poses_batch=poses_batch,
                                 lr=lr,
                                 dropout_keep=self.config.dropout_keep)
    _, loss, grad_norm, learning_rate, logits_pose_norm, logits_label_norm, grads_vars = sess.run(
      [self.train_op, self.loss, self.grad_norm, self.learning_rate_placeholder, self.logits_pose_norm, self.logits_label_norm, self.grads_vars], feed_dict=feed)
    return loss, grad_norm, learning_rate, logits_pose_norm, logits_label_norm, grads_vars

  def evaluate_on_batch(self, sess, input_batch, labels_batch, poses_batch):
    feed = self.create_feed_dict(is_training=False,
                                 input_batch=input_batch,
                                 labels_batch=labels_batch,
                                 poses_batch=poses_batch)
    pred, loss = sess.run([self.pred, self.loss], feed_dict=feed)  # pick the class that has highest probability
    delta_az, delta_el, delta_di, vox_pred = pred
    poses_label = poses_batch[:, 1:] - poses_batch[:, :-1]
    delta_az_label = poses_label[:, :, 0]
    delta_el_label = poses_label[:, :, 1]
    delta_di_label = poses_label[:, :, 2]
    az_rmse = np.mean(np.linalg.norm(delta_az_label - delta_az, axis=-1))
    el_rmse = np.mean(np.linalg.norm(delta_el_label - delta_el, axis=-1))
    di_rmse = np.mean(np.linalg.norm(delta_di_label - delta_di, axis=-1))

    thresh = self.config.TEST.VOXEL_THRESH[0]
    iou = lib.voxel.evaluate_voxel_prediction(vox_pred, labels_batch, thresh)
    return az_rmse, el_rmse, di_rmse, iou

  def add_placeholders(self):
    self.is_training_placeholder = tf.placeholder(tf.bool, shape=())
    self.input_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.CONST.N_VIEWS, self.config.CONST.IMG_H, self.config.CONST.IMG_W, 3))
    self.labels_placeholder = tf.placeholder(tf.int32, shape=(None, self.config.CONST.N_VOX, self.config.CONST.N_VOX, self.config.CONST.N_VOX))
    self.poses_placeholder = tf.placeholder(tf.float32, shape=(None, self.config.CONST.N_VIEWS, 3))
    self.dropout_keep_placeholder = tf.placeholder(tf.float32)
    self.learning_rate_placeholder = tf.placeholder(tf.float32)

  def create_feed_dict(self, is_training, input_batch, labels_batch=None, poses_batch=None, lr=None, dropout_keep=1.0):
    feed_dict = {self.is_training_placeholder: is_training,
                 self.input_placeholder: input_batch,
                 self.poses_placeholder: poses_batch,
                 self.dropout_keep_placeholder: dropout_keep}

    if not lr is None:
      feed_dict[self.learning_rate_placeholder] = lr
    if not labels_batch is None:
      feed_dict[self.labels_placeholder] = labels_batch

    return feed_dict

  def add_logit_op(self):
    # self.input_placeholder: (None, n_timesteps, height, width, n_channels)

    with tf.variable_scope("R2N2_logit"):

      # reuse = tf.logical_not(self.is_training_placeholder)
      # reshape to merge n_batches, CONST.N_VIEWS
      # because conv2d() function only takes 4D tensor, for 2D convolution
      self.input_placeholder.get_shape()
      input = tf.reshape(self.input_placeholder, shape=(-1, self.config.CONST.IMG_H, self.config.CONST.IMG_W, 3))
      # input = tf.Print(input, [tf.reduce_min(input), tf.reduce_max(input), input], message="input")

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
      conv5_flattened = tf.Print(conv5_flattened, [tf.reduce_min(conv5_flattened), tf.reduce_max(conv5_flattened), conv5_flattened], message="conv5_flattened")

      n_fc_outputs = 1024
      fc = tf.contrib.layers.fully_connected(inputs=conv5_flattened, num_outputs=n_fc_outputs,
                                             activation_fn=None, scope="fc1", reuse=False)
      fc = tf.contrib.layers.batch_norm(fc, center=True, scale=True, is_training=self.is_training_placeholder,
                                        scope='fc_batch_norm')
      fc = tf.nn.relu(fc, name='relu')

      # reshape back to (batch_size, CONST.N_VIEWS, n_fc_outputs)
      fc = tf.reshape(fc, shape=(-1, self.config.CONST.N_VIEWS, n_fc_outputs))
      fc = tf.Print(fc, [tf.reduce_min(fc), tf.reduce_max(fc), fc], message="fc")

      # predict pose delta
      states_concat = tf.concat([fc[:, 1:], fc[:, :-1]], axis=-1)
      states_concat_size = states_concat.get_shape()[-1].value
      W_dfc1 = tf.get_variable("W_dfc1", shape=(states_concat_size, 128), initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
      fc_delta = tf.einsum('ijk,kl->ijl', states_concat, W_dfc1)
      fc_delta = tf.contrib.layers.batch_norm(fc_delta, center=True, scale=True, is_training=self.is_training_placeholder, scope='fc_delta_batch_norm')
      fc_delta = tf.nn.relu(fc_delta)
      fc_delta = tf.Print(fc_delta, [tf.reduce_min(fc_delta), tf.reduce_max(fc_delta), fc_delta], message="fc_delta")
      W_dfc2 = tf.get_variable("W_dfc2", shape=(128, 3), initializer=tf.contrib.layers.xavier_initializer(), dtype=np.float32)
      b_dfc2 = tf.get_variable("b_dfc2", shape=3, dtype=np.float32)
      delta_poses = tf.einsum('ijk,kl->ijl', fc_delta, W_dfc2) + b_dfc2
      delta_az = delta_poses[:, :, 0]
      delta_el = delta_poses[:, :, 1]
      delta_di = delta_poses[:, :, 2]
      delta_az = tf.Print(delta_az, [tf.reduce_min(delta_az), tf.reduce_max(delta_az), tf.reduce_mean(delta_az), delta_az], message="delta_az")
      delta_el = tf.Print(delta_el, [tf.reduce_min(delta_el), tf.reduce_max(delta_el), tf.reduce_mean(delta_el), delta_el], message="delta_el")
      delta_di = tf.Print(delta_di, [tf.reduce_min(delta_di), tf.reduce_max(delta_di), tf.reduce_mean(delta_di), delta_di], message="delta_di")

      batch_size = tf.shape(delta_poses)[0] # get batch size as tensor
      # append [0, 0, 0] as last delta pose
      delta_poses_in = tf.concat([delta_poses, tf.zeros((batch_size, 1, 3))], axis=1)
      # augment fully connected output with delta poses info
      fc = tf.concat([fc, delta_poses_in], axis=-1)

      # 3D GRU
      grid_state_size = (4, 4, 4, 128)
      cell = GRU3dCell(fc.shape[-1].value, grid_state_size)
      states, h = tf.nn.dynamic_rnn(cell, fc, dtype=tf.float32)
      shape = [-1] + list(grid_state_size)
      h = tf.reshape(h, shape=shape) # reshape back to 3d
      h = tf.Print(h, [tf.reduce_min(h), tf.reduce_max(h), h], message="3D GRU output")

      # deconvolutional layers
      # 1st deconv layer
      h = models.unpool_3d.unpool_3d_zero_filled(h)
      deconv11 = tf.layers.conv3d(h, filters=128, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                  activation=tf.nn.relu, use_bias=False, name="deconv11", reuse=False)
      deconv12 = tf.layers.conv3d(deconv11, filters=128, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                  activation=tf.nn.relu, use_bias=False, name="deconv12", reuse=False)
      deconv1_res = tf.layers.conv3d(h, filters=128, kernel_size=[1, 1, 1], strides=(1, 1, 1), padding='same',
                                     activation=tf.nn.relu, use_bias=False, name="deconv1_res", reuse=False)
      deconv1 = deconv12 + deconv1_res

      # 2nd deconv layer
      deconv1 = models.unpool_3d.unpool_3d_zero_filled(deconv1)
      deconv21 = tf.layers.conv3d(deconv1, filters=128, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                  activation=tf.nn.relu, use_bias=False, name="deconv21", reuse=False)
      deconv22 = tf.layers.conv3d(deconv21, filters=128, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                  activation=tf.nn.relu, use_bias=False, name="deconv22", reuse=False)
      deconv2_res = tf.layers.conv3d(deconv1, filters=128, kernel_size=[1, 1, 1], strides=(1, 1, 1), padding='same',
                                     activation=tf.nn.relu, use_bias=False, name="deconv2_res", reuse=False)
      deconv2 = deconv22 + deconv2_res

      # 3rd deconv layer
      deconv2 = models.unpool_3d.unpool_3d_zero_filled(deconv2)
      deconv31 = tf.layers.conv3d(deconv2, filters=64, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                  activation=tf.nn.relu, use_bias=False, name="deconv31", reuse=False)
      deconv32 = tf.layers.conv3d(deconv31, filters=64, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                  activation=tf.nn.relu, use_bias=False, name="deconv32", reuse=False)
      deconv3_res = tf.layers.conv3d(deconv2, filters=64, kernel_size=[1, 1, 1], strides=(1, 1, 1), padding='same',
                                     activation=tf.nn.relu, use_bias=False, name="deconv3_res", reuse=False)
      deconv3 = deconv32 + deconv3_res

      # 4th deconv layer
      deconv41 = tf.layers.conv3d(deconv3, filters=32, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                  activation=tf.nn.relu, use_bias=False, name="deconv41", reuse=False)
      deconv42 = tf.layers.conv3d(deconv41, filters=32, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                  activation=tf.nn.relu, use_bias=False, name="deconv42", reuse=False)
      deconv4_res = tf.layers.conv3d(deconv3, filters=32, kernel_size=[1, 1, 1], strides=(1, 1, 1), padding='same',
                                     activation=tf.nn.relu, use_bias=False, name="deconv4_res", reuse=False)
      deconv4 = deconv42 + deconv4_res

      # final deconv layer
      deconv5 = tf.layers.conv3d(deconv4, filters=2, kernel_size=[3, 3, 3], strides=(1, 1, 1), padding='same',
                                 activation=None, use_bias=False, name="deconv4", reuse=False)
      deconv5 = tf.Print(deconv5, [tf.reduce_min(deconv5), tf.reduce_max(deconv5), deconv5], message="deconv5")

    return delta_az, delta_el, delta_di, deconv5

  def add_prediction_op(self, logits):
    return logits

  def add_loss_op(self, logits):
    delta_az, delta_el, delta_di, deconv = logits
    self.logits_pose_norm = tf.sqrt(tf.reduce_mean(tf.square([delta_az, delta_el, delta_di])))
    self.logits_label_norm = tf.sqrt(tf.reduce_mean(tf.square(deconv)))
    self.labels_placeholder = tf.Print(self.labels_placeholder, [self.labels_placeholder], message="labels")

    cross_entropy_label = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=deconv, labels=self.labels_placeholder)
    loss_label = tf.reduce_mean(cross_entropy_label)
    loss_label = tf.Print(loss_label, [loss_label], message="loss_label")

    delta_poses_label = self.poses_placeholder[:,1:] - self.poses_placeholder[:,:-1]
    delta_az_label = delta_poses_label[:, :, 0]
    delta_el_label = delta_poses_label[:, :, 1]
    delta_di_label = delta_poses_label[:, :, 2]
    delta_az_label = tf.Print(delta_az_label, [tf.reduce_min(delta_az_label), tf.reduce_max(delta_az_label)], message="delta_az label")
    delta_el_label = tf.Print(delta_el_label, [tf.reduce_min(delta_el_label), tf.reduce_max(delta_el_label)], message="delta_el label")
    delta_di_label = tf.Print(delta_di_label, [tf.reduce_min(delta_di_label), tf.reduce_max(delta_di_label)], message="delta_di label")

    rmse_az = tf.sqrt(tf.reduce_mean(tf.square(delta_az - delta_az_label)))
    rmse_el = tf.sqrt(tf.reduce_mean(tf.square(delta_el - delta_el_label)))
    rmse_di = tf.sqrt(tf.reduce_mean(tf.square(delta_di - delta_di_label)))
    rmse_az = tf.Print(rmse_az, [rmse_az], message="rmse_az")
    rmse_el = tf.Print(rmse_el, [rmse_el], message="rmse_el")
    rmse_di = tf.Print(rmse_di, [rmse_di], message="rmse_di")

    loss = tf.reduce_mean([rmse_az, rmse_el, rmse_di]) + loss_label
    return loss

  def add_training_op(self, loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_placeholder)
      grads_vars = optimizer.compute_gradients(loss)
      filtered_grads_vars = []
      for grad, var in grads_vars:
        if not grad is None:
          filtered_grads_vars.append((grad, var))
      grads = [pair[0] for pair in filtered_grads_vars]
      self.grad_norm = tf.global_norm(grads)
      self.grads_vars = filtered_grads_vars
      train_op = optimizer.apply_gradients(filtered_grads_vars)
    return train_op
