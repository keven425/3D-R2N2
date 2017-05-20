import os
import numpy as np
import datetime as dt

from lib.config import cfg



class Model(object):

    def __init__(self, random_seed=dt.datetime.now().microsecond):
        self.rng = np.random.RandomState(random_seed)

        self.batch_size = cfg.CONST.BATCH_SIZE
        self.img_w = cfg.CONST.IMG_W
        self.img_h = cfg.CONST.IMG_H
        self.n_vox = cfg.CONST.N_VOX

        self.build()

    ###################### Build the model ##############################

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.

        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building and will be fed data during
        training.

        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, input_batch, labels_batch=None, dropout=1):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_logit_op(self):
        raise NotImplementedError

    def add_prediction_op(self, logit):
        """Implements the core of the model that transforms a batch of input data into predictions.

        Returns:
            pred: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, logit):
        """Adds Ops for the loss function to the computational graph.

        Args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        """Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See

        https://www.tensorflow.org/versions/r0.7/api_docs/python/train.html#Optimizer

        for more information.

        Args:
            loss: Loss tensor (a scalar).
        Returns:
            train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, sess, inputs_batch, labels_batch, lr):
        """Perform one step of gradient descent on the provided batch of data.

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
            labels_batch: np.ndarray of shape (n_samples, n_classes)
        Returns:
            loss: loss over the batch (a scalar)
        """
        raise NotImplementedError
        # feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch)
        # _, loss = sess.run([self.train_op, self.loss], feed_dict=feed)
        # return loss

    def evaluate_on_batch(self, sess, inputs_batch):
        """Make predictions for the provided batch of data

        Args:
            sess: tf.Session()
            input_batch: np.ndarray of shape (n_samples, n_features)
        Returns:
            predictions: np.ndarray of shape (n_samples, n_classes)
        """
        raise NotImplementedError
        # feed = self.create_feed_dict(inputs_batch)
        # predictions = sess.run(self.pred, feed_dict=feed)
        # return predictions

    def build(self):
        self.add_placeholders()
        self.logit = self.add_logit_op()
        self.pred = self.add_prediction_op(self.logit)
        self.loss = self.add_loss_op(self.logit)
        self.train_op = self.add_training_op(self.loss)

    ################## Run training and evaluation #########################

    def preprocess_sequence_data(self, input_list):
        """Preprocess sequence data for the model. For example, apply padding.

        Args:
            examples: A list of inputs (intput_text1, intput_text2, label) that are turned into indices.
        Returns:
            A new list of vectorized input tuples appropriate for the model.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def consolidate_predictions(self, data_raw, data, preds):
        """
        apply masking to predictions
        Then group the input_text_2, input_text_2, labels, predicted_labels, in that order

        Args:
            data_raw: raw text data read from file, a list of tuples (input1, input2, label)
            data: processed data that has been turned into indices. a list of tuples (input1, input2, label)
            preds: predictions made by the model. a list of vectors with size num_classes
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def evaluate(self, sess, examples, examples_raw, best_score, train_data=False):
        """Evaluates model performance on @examples.

        This function uses the model to predict labels for @examples and constructs a confusion matrix.

        Args:
            sess: the current TensorFlow session.
            examples: A list of vectorized input/output pairs.
            examples: A list of the original input/output sequence pairs.
        Returns:
            The F1 score for predicting tokens as named entities.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def save(self, saver, sess, best=False):
        if saver:
            path = os.path.join(cfg.DIR.OUT_PATH + '/model.ckpt')
            print("Saving model in %s", path)
            dirpath = os.path.dirname(path)
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)
            saver.save(sess, path)
