#!/usr/bin/env python
import sys
if (sys.version_info < (3, 0)):
    raise Exception("Please follow the installation instruction on 'https://github.com/keven425/3D-R2N2'")

import numpy as np
import argparse
import pprint
import logging
import logging.handlers
import time
import multiprocessing as mp

# Theano
from lib.config import cfg, cfg_from_file, cfg_from_list
import models.train
import models.test


def parse_args():
    parser = argparse.ArgumentParser(description='Main 3Deverything train/test file')

    parser.add_argument(
        '--cfg',
        dest='cfg_files',
        action='append',
        help='optional config file',
        default=None,
        type=str)
    parser.add_argument(
        '--name',
        dest='name',
        help='name of the run',
        default=None,
        type=str)
    parser.add_argument(
        '--rand',
        dest='randomize',
        help='randomize (do not use a fixed seed)',
        action='store_true')
    parser.add_argument(
        '--test',
        dest='test',
        help='randomize (do not use a fixed seed)',
        default=False,
        action='store_true')
    parser.add_argument(
        '--batch-size',
        dest='batch_size',
        help='batch size',
        default=cfg.CONST.BATCH_SIZE,
        type=int)
    parser.add_argument(
      '--views',
      dest='n_views',
      help='n views to use',
      default=cfg.CONST.N_VIEWS,
      type=int)
    parser.add_argument(
        '--sample-every',
        dest='sample_every',
        help='sample every n frames',
        default=cfg.sample_every,
        type=int)
    parser.add_argument(
        '--iter',
        dest='iter',
        help='number of iterations',
        default=cfg.TRAIN.NUM_ITERATION,
        type=int)
    parser.add_argument(
        '--dataset',
        dest='dataset',
        help='dataset config file',
        default=None,
        type=str)
    parser.add_argument(
        '--set',
        dest='set_cfgs',
        help='set config keys',
        default=None,
        nargs=argparse.REMAINDER)
    parser.add_argument(
        '--weights',
        dest='weights',
        help='Initialize network from the weights file',
        default=None)
    parser.add_argument(
        '--init-iter',
        dest='init_iter',
        help='Start from the specified iteration',
        default=cfg.TRAIN.INITIAL_ITERATION)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.cfg_files is not None:
        for cfg_file in args.cfg_files:
            cfg_from_file(cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)
    if not args.randomize:
        np.random.seed(cfg.CONST.RNG_SEED)

    if args.batch_size is not None:
        cfg_from_list(['CONST.BATCH_SIZE', args.batch_size])
    if args.n_views is not None:
        cfg_from_list(['CONST.N_VIEWS', args.n_views])
    if args.sample_every is not None:
        cfg_from_list(['sample_every', args.sample_every])
    if args.iter is not None:
        cfg_from_list(['TRAIN.NUM_ITERATION', args.iter])
    if args.dataset is not None:
        cfg_from_list(['DATASET', args.dataset])
    if args.weights is not None:
        cfg_from_list(['DIR.WEIGHTS_PATH', args.weights])

    # set output path based on config params and name
    out_path = './output/'
    if args.name:
        out_path += args.name + '_'
    out_path += 'batch' + str(cfg.CONST.BATCH_SIZE) + '_' \
              + 'niter' + str(cfg.TRAIN.NUM_ITERATION) + '_' \
              + str(int(time.time()))
    cfg.DIR.OUT_PATH = out_path

    print('Using config:')
    pprint.pprint(cfg)

    if not args.test:
        models.train.train_net()
    else:
        models.test.test_net()


if __name__ == '__main__':
    mp.log_to_stderr()
    logger = mp.get_logger()
    logger.setLevel(logging.INFO)
    main()
