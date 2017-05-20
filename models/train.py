import tensorflow as tf
import inspect
from multiprocessing import Queue
import os
from datetime import datetime
import numpy as np

from models.R2N2Model import R2N2Model

from lib.config import cfg
from lib.data_io import category_model_id_pair
from lib.data_process import kill_processes, make_data_processes
from lib.utils import Timer

# Define globally accessible queues, will be used for clean exit when force
# interrupted.
train_queue, val_queue, train_processes, val_processes = None, None, None, None


def cleanup_handle(func):
    '''Cleanup the data processes before exiting the program'''

    def func_wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            print('Wait until the dataprocesses to end')
            kill_processes(train_queue, train_processes)
            kill_processes(val_queue, val_processes)
            raise

    return func_wrapper


@cleanup_handle
def train_net():
    '''Main training function'''

    # Check that single view reconstruction net is not used for multi view
    # reconstruction.
    # if net.is_x_tensor4 and cfg.CONST.N_VIEWS > 1:
    #     raise ValueError('Do not set the config.CONST.N_VIEWS > 1 when using' \
    #                      'single-view reconstruction network')

    # Prefetching data processes
    #
    # Create worker and data queue for data processing. For training data, use
    # multiple processes to speed up the loading. For validation data, use 1
    # since the queue will be popped every TRAIN.NUM_VALIDATION_ITERATIONS.
    global train_queue, val_queue, train_processes, val_processes
    train_queue = Queue(cfg.QUEUE_SIZE)
    val_queue = Queue(cfg.QUEUE_SIZE)

    train_processes = make_data_processes(
        train_queue,
        category_model_id_pair(dataset_portion=cfg.TRAIN.DATASET_PORTION),
        cfg.TRAIN.NUM_WORKER,
        repeat=True)
    val_processes = make_data_processes(
        val_queue,
        category_model_id_pair(dataset_portion=cfg.TEST.DATASET_PORTION),
        1,
        repeat=True,
        train=False)

    # Train the network
    train(train_queue, val_queue)

    # Cleanup the processes and the queue.
    kill_processes(train_queue, train_processes)
    kill_processes(val_queue, val_processes)


def train(train_queue, val_queue=None):
    ''' Given data queues, train the network '''

    with tf.Graph().as_default():

        model = R2N2Model(cfg)

        print('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
        saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)

        with tf.Session() as session:

            if cfg.DIR.WEIGHTS_PATH:
                # load pre-trained weights
                print('Restoring model from: ' + cfg.DIR.WEIGHTS_PATH)
                saver.restore(session, cfg.DIR.WEIGHTS_PATH)
                variables_to_init = []
                all_variables = tf.global_variables()
                for var in all_variables:
                    if not session.run(tf.is_variable_initialized(var)):
                        variables_to_init.append(var)
                init = tf.variables_initializer(variables_to_init)
            else:
                init = tf.global_variables_initializer()
            session.run(init)

            # Parameter directory
            save_dir = os.path.join(cfg.DIR.OUT_PATH)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # Timer for the training op and parallel data loading op.
            train_timer = Timer()
            data_timer = Timer()
            training_losses = []

            start_iter = 0
            # Resume training
            # if cfg.TRAIN.RESUME_TRAIN:
            #     self.net.load(cfg.CONST.WEIGHTS)
            #     start_iter = cfg.TRAIN.INITIAL_ITERATION

            # Setup learning rates
            lr = cfg.TRAIN.DEFAULT_LEARNING_RATE
            lr_steps = [int(k) for k in cfg.TRAIN.LEARNING_RATES.keys()]

            print('Set the learning rate to %f.' % lr)

            # Main training loop
            for train_ind in range(start_iter, cfg.TRAIN.NUM_ITERATION + 1):
                data_timer.tic()
                batch_img, batch_voxel = train_queue.get()
                data_timer.toc()

                # if self.net.is_x_tensor4:
                #     batch_img = batch_img[0]

                # Decrease learning rate at certain points
                if train_ind in lr_steps:
                    # edict only takes string for key. Hacky way
                    lr = np.float(cfg.TRAIN.LEARNING_RATES[str(train_ind)])

                # Apply one gradient step
                train_timer.tic()
                loss, grad_norm, learning_rate, logits_norm, grads_vars = model.train_on_batch(session, batch_img, batch_voxel, lr)
                train_timer.toc()
                print('loss: %f, gradnorm: %f, lr: %f, logitsnorm: %f' % (loss, grad_norm, learning_rate, logits_norm))

                training_losses.append(loss)

                # Debugging modules
                #
                # Print status, run validation, check divergence, and save model.
                if train_ind % cfg.TRAIN.PRINT_FREQ == 0:
                    # Print the current loss
                    print('%s Iter: %d Loss: %f' % (datetime.now(), train_ind, loss))

                if train_ind % cfg.TRAIN.VALIDATION_FREQ == 0 and val_queue is not None:
                    # Print test loss and params to check convergence every N iterations
                    ious = []
                    for i in range(cfg.TRAIN.NUM_VALIDATION_ITERATIONS):
                        batch_img, batch_voxel = val_queue.get()
                        iou = model.evaluate_on_batch(session, batch_img, batch_voxel)
                        ious.append(np.mean(iou))
                    print('%s Validation IoU: %f' % (datetime.now(), np.mean(ious)))

                if train_ind % cfg.TRAIN.SAVE_FREQ == 0 and not train_ind == 0:
                    model.save(saver, session)


def main():
    '''Test function'''
    cfg.DATASET = '/cvgl/group/ShapeNet/ShapeNetCore.v1/cat1000.json'
    cfg.CONST.RECNET = 'rec_net'
    cfg.TRAIN.DATASET_PORTION = [0, 0.8]
    train_net()


if __name__ == '__main__':
    main()
