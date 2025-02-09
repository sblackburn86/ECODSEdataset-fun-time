#!/usr/bin/env python

import argparse
import datetime
import logging
import os
import sys

import numpy as np
import tensorflow as tf

from mlflow import log_metric, log_param
from orion.client import report_results
from yaml import load
import ecodse_funtime_alpha.data as data
import ecodse_funtime_alpha.models as models
import ecodse_funtime_alpha.train as train

tf.compat.v1.enable_eager_execution()
logger = logging.getLogger(__name__)


def get_args(args):
    """
    read and parse the arguments

    Parameters
    ----------
    args : sys.argv
        arguments specified by the user

    Returns
    -------
    ArgumentParser object
       object containing the input arguments
    """
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--config',
                           required=True,
                           help='config file with hyper parameters - in yaml format')

    def_log = None
    argparser.add_argument('--log',
                           default=def_log,
                           help=f'log to this file (default {def_log}: log to screen)')

    def_checkpointfrq = 'epoch'
    argparser.add_argument('--ckptfreq',
                           default=def_checkpointfrq,
                           help=f'how many examples passed in training before saving (default is once per epoch)')

    def_checkpointpath = 'saved_model/cp-{epoch:04d}.ckpt'
    argparser.add_argument('--ckptpath',
                           type=str,
                           default=def_checkpointpath,
                           help=f'checkpoint path (default {def_checkpointpath})')

    def_impath = '../../rainforest/fixed-train-jpg/'
    argparser.add_argument('--imagepath',
                           default=def_impath,
                           help=f'path to image directory (default {def_impath})')

    def_labelpath = '../../rainforest/train_v3.csv'
    argparser.add_argument('--labelpath',
                           default=def_labelpath,
                           help=f'path to csv file for labels (defautlt {def_labelpath})')

    def_seed = 0
    argparser.add_argument('-s',
                           '--seed',
                           default=def_seed,
                           type=int,
                           help=f'Set random seed to this number (default {def_seed})')

    def_dataseed = 0
    argparser.add_argument('-ds',
                           '--dataseed',
                           default=def_dataseed,
                           type=int,
                           help=f'Set random seed for data split into train / valid / test sets')

    def_kernels = 4
    argparser.add_argument('-k',
                           '--kernels',
                           default=def_kernels,
                           type=int,
                           help=f'Number of kernels in the CNN (default {def_kernels})')
    def_ksize = 2
    argparser.add_argument('-ks',
                           '--ksize',
                           default=def_ksize,
                           type=int,
                           help=f'Size of kernels in CNN (default {def_ksize})')

    def_f1 = 64
    argparser.add_argument('-f1',
                           '--filter1',
                           default=def_f1,
                           type=int,
                           help=f'Number of kernels in the first VGG block (default {def_f1})')

    def_k1 = 3
    argparser.add_argument('-ks1',
                           '--ksize1',
                           default=def_k1,
                           type=int,
                           help=f'Size of kernels in the first VGG block (default {def_k1})')

    def_s1 = 1
    argparser.add_argument('-s1',
                           '--stride1',
                           default=def_s1,
                           type=int,
                           help=f'Stride in the first VGG block (default {def_s1})')

    def_p1 = 2
    argparser.add_argument('-p1',
                           '--pool1',
                           default=def_p1,
                           type=int,
                           help=f'Max pooling size in the first VGG block (default {def_p1})')

    def_sp1 = 2
    argparser.add_argument('-sp1',
                           '--stridepool1',
                           default=def_sp1,
                           type=int,
                           help=f'Pool stride in the first VGG block (default {def_sp1})')

    def_f2 = 128
    argparser.add_argument('-f2',
                           '--filter2',
                           default=def_f2,
                           type=int,
                           help=f'Number of kernels in the second VGG block (default {def_f2})')

    def_k2 = 3
    argparser.add_argument('-ks2',
                           '--ksize2',
                           default=def_k2,
                           type=int,
                           help=f'Size of kernels in the second VGG block (default {def_k2})')

    def_s2 = 1
    argparser.add_argument('-s2',
                           '--stride2',
                           default=def_s2,
                           type=int,
                           help=f'Stride in the second VGG block (default {def_s2})')

    def_p2 = 2
    argparser.add_argument('-p2',
                           '--pool2',
                           default=def_p2,
                           type=int,
                           help=f'Max pooling size in the second VGG block (default {def_p2})')

    def_sp2 = 2
    argparser.add_argument('-sp2',
                           '--stridepool2',
                           default=def_sp2,
                           type=int,
                           help=f'Pool stride in the second VGG block (default {def_sp2})')

    def_f3 = 256
    argparser.add_argument('-f3',
                           '--filter3',
                           default=def_f3,
                           type=int,
                           help=f'Number of kernels in the 3rd VGG block (default {def_f3})')

    def_k3 = 3
    argparser.add_argument('-ks3',
                           '--ksize3',
                           default=def_k3,
                           type=int,
                           help=f'Size of kernels in the 3rd VGG block (default {def_k3})')

    def_s3 = 1
    argparser.add_argument('-s3',
                           '--stride3',
                           default=def_s3,
                           type=int,
                           help=f'Stride in the 3rd VGG block (default {def_s3})')

    def_p3 = 2
    argparser.add_argument('-p3',
                           '--pool3',
                           default=def_p3,
                           type=int,
                           help=f'Max pooling size in the 3rd VGG block (default {def_p3})')

    def_sp3 = 2
    argparser.add_argument('-sp3',
                           '--stridepool3',
                           default=def_sp3,
                           type=int,
                           help=f'Pool stride in the 3rd VGG block (default {def_sp3})')

    def_f4 = 512
    argparser.add_argument('-f4',
                           '--filter4',
                           default=def_f4,
                           type=int,
                           help=f'Number of kernels in the 4th VGG block (default {def_f4})')

    def_k4 = 3
    argparser.add_argument('-ks4',
                           '--ksize4',
                           default=def_k4,
                           type=int,
                           help=f'Size of kernels in the 4th VGG block (default {def_k4})')

    def_s4 = 1
    argparser.add_argument('-s4',
                           '--stride4',
                           default=def_s4,
                           type=int,
                           help=f'Stride in the 4th VGG block (default {def_s4})')

    def_p4 = 2
    argparser.add_argument('-p4',
                           '--pool4',
                           default=def_p4,
                           type=int,
                           help=f'Max pooling size in the 4th VGG block (default {def_p4})')

    def_sp4 = 2
    argparser.add_argument('-sp4',
                           '--stridepool4',
                           default=def_sp4,
                           type=int,
                           help=f'Pool stride in the 4th VGG block (default {def_sp4})')

    def_f5 = 512
    argparser.add_argument('-f5',
                           '--filter5',
                           default=def_f5,
                           type=int,
                           help=f'Number of kernels in the 5th VGG block (default {def_f5})')

    def_k5 = 3
    argparser.add_argument('-ks5',
                           '--ksize5',
                           default=def_k5,
                           type=int,
                           help=f'Size of kernels in the 5th VGG block (default {def_k5})')

    def_s5 = 1
    argparser.add_argument('-s5',
                           '--stride5',
                           default=def_s5,
                           type=int,
                           help=f'Stride in the 5th VGG block (default {def_s5})')

    def_p5 = 2
    argparser.add_argument('-p5',
                           '--pool5',
                           default=def_p5,
                           type=int,
                           help=f'Max pooling size in the 5th VGG block (default {def_p5})')

    def_sp5 = 2
    argparser.add_argument('-sp5',
                           '--stridepool5',
                           default=def_sp5,
                           type=int,
                           help=f'Pool stride in the 5th VGG block (default {def_sp5})')

    def_dense1 = 4096
    argparser.add_argument('-d1',
                           '--dense1',
                           default=def_dense1,
                           type=int,
                           help=f'Size of 1st dense layer in the VGG network (default {def_dense1})')

    def_dense2 = 4096
    argparser.add_argument('-d2',
                           '--dense2',
                           default=def_dense2,
                           type=int,
                           help=f'Size of 2nd dense layer in the VGG network (default {def_dense2})')

    def_augmentpolicy = None
    argparser.add_argument('-da',
                           '--dataaugment',
                           default=def_augmentpolicy,
                           type=str,
                           help=f'Data augmentation scheme based on AutoAugment Implemented options are \n cifar10\n imagenet \n svhn \n(default {def_augmentpolicy})')

    def_lr = 0.1
    argparser.add_argument('-l',
                           '--lr',
                           default=def_lr,
                           type=float,
                           help=f'Learning rate (default {def_lr})')

    def_nepoch = 1
    argparser.add_argument('-n',
                           '--nepoch',
                           default=def_nepoch,
                           type=int,
                           help=f'Number of epoch for training (default {def_nepoch})')

    def_patience = 10
    argparser.add_argument('-p',
                           '--patience',
                           default=def_patience,
                           type=int,
                           help=f'Number of epoch without improvements before stopping training (default {def_patience})')

    def_batch = 4
    argparser.add_argument('-b',
                           '--batchsize',
                           default=def_batch,
                           type=int,
                           help=f'batch size (default {def_batch})')

    args = argparser.parse_args(args)
    return args


class ModelCallback(tf.keras.callbacks.Callback):
    """callback method logging validation accuracy in MLflow, Orion. Also implements early stopping

    Attributes
    ----------
    best_valid_acc : float
        best validation accuracy
    best_weights : tf.keras.Model weights
        best weights for the model when reaching best_valid_acc
    not_improving_since : int
        how many epochs since the validation accuracy improved
    patience: int
        how many epochs the validation accuracy does not improve before stopping the model
    stopped_epoch : int
        at which epoch the model was (early) stopped

    Methods
    -------
    on_train_begin:
        operations to perform when training begins
    on_train_epoch_end:
        operations to perform when an epoch of training is complete
    on_train_end:
        operations to perform when training is complete

    """

    def __init__(self, patience=10):
        """class constructor

        Parameters
        ----------
        patience : int, optional
            number of epochs to train without improvement before early stopping, by default 10
        """
        super(ModelCallback, self).__init__()

        self.best_valid_acc = None
        self.not_improving_since = 0
        self.patience = patience
        self.best_weights = None
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        """operations when starting training

        Parameters
        ----------
        logs : dict, optional
            dictionary containing the model evaluation metrics, by default None
        """
        self.start_time = datetime.datetime.now()
        print(f"Starting training at {self.start_time}")

    def on_epoch_end(self, epoch, logs=None):
        """operations to complete after an epoch is complete

        Parameters
        ----------
        epoch : int
            index of epoch
        logs : dict, optional
            dictionary containing the model evaluation metrics, by default None
        """
        logging.info(f"For epoch {epoch}, training loss / accuracy is {'{:7.2f}'.format(logs['loss'])} / {'{:7.2f}'.format(logs['acc'])} ")
        logging.info(f"For epoch {epoch}, validation loss / accuracy is {'{:7.2f}'.format(logs['val_loss'])} / {'{:7.2f}'.format(logs['val_acc'])} ")
        log_metric("training_loss", logs["loss"], step=epoch)
        log_metric("training_acc", logs["acc"], step=epoch)
        log_metric("valid_loss", logs["val_loss"], step=epoch)
        log_metric("valid_acc", logs["val_acc"], step=epoch)

        if self.best_valid_acc is None or logs["val_acc"] > self.best_valid_acc:
            self.best_valid_acc = logs["val_acc"]
            self.best_weights = self.model.get_weights()
            self.not_improving_since = 0
        else:
            self.not_improving_since += 1

        logging.info('\ndone epoch {} => loss {} - validation accuracy {} (not improving'
                     ' since {} epoch)'.format(epoch, logs['loss'], logs['val_acc'], self.not_improving_since))

        if self.not_improving_since >= self.patience:
            self.stopped_epoch = epoch
            self.model.stop_training = True
            logging.info("Restoring model weights from the end of the best epoch")
            self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        """operations to perform after training is complete

        Parameters
        ----------
        logs : dict, optional
            dictionary containing the model evaluation metrics, by default None
        """
        self.end_time = datetime.datetime.now()
        logging.info(f"Ending training at {self.end_time}")
        logging.info(f"Training duration: {self.end_time - self.start_time}")
        if self.stopped_epoch > 0:
            logging.info(f"Early stopping at epoch {self.stopped_epoch}")
        log_metric("best_valid_acc", self.best_valid_acc)
        report_results([dict(
            name="valid_acc",
            type="objective",
            value=-self.best_valid_acc
        )])


if __name__ == "__main__":
    args = get_args(sys.argv[1:])

    assert args.ckptfreq == 'epoch' or isinstance(args.ckptfreq, int)

    logging.basicConfig(level=logging.INFO)

    # will log to a file if provided (useful for orion on cluster)
    if args.log is not None:
        handler = logging.handlers.WatchedFileHandler(args.log)
        formatter = logging.Formatter(logging.BASIC_FORMAT)
        handler.setFormatter(formatter)
        root = logging.getLogger()
        root.setLevel(logging.INFO)
        root.addHandler(handler)

    with open(args.config, 'r') as stream:
        # hp stands from hyper parameters
        hp = load(stream)

    tf.compat.v1.random.set_random_seed(args.seed)
    np.random.seed(args.seed)

    # split data into train/valid/test set
    if not os.path.exists(os.path.join(args.imagepath, "splits/")):
        os.makedirs(os.path.join(args.imagepath, "splits/"))

    train_csv, valid_csv, test_csv = data.split_train_val_test(args.labelpath, os.path.join(args.imagepath, "splits"),
                                                               train_size=0.6, val_size=0.2, seed=args.dataseed)

    train_dataset = data.get_dataset(args.imagepath, train_csv)
    valid_dataset = data.get_dataset(args.imagepath, valid_csv)
    test_dataset = data.get_dataset(args.imagepath, test_csv)

    log_param("random_seed", args.seed)
    log_param("filter1", hp['filter1'])
    log_param("filter2", hp['filter2'])
    log_param("filter3", hp['filter3'])
    log_param("filter4", hp['filter4'])
    log_param("filter5", hp['filter5'])
    log_param("kernel_size1", hp['ksize1'])
    log_param("kernel_size2", hp['ksize2'])
    log_param("kernel_size3", hp['ksize3'])
    log_param("kernel_size4", hp['ksize4'])
    log_param("kernel_size5", hp['ksize5'])
    log_param("stride1", hp['stride1'])
    log_param("stride2", hp['stride2'])
    log_param("stride3", hp['stride3'])
    log_param("stride4", hp['stride4'])
    log_param("stride5", hp['stride5'])
    log_param("pool_size1", hp['pool1'])
    log_param("pool_size2", hp['pool2'])
    log_param("pool_size3", hp['pool3'])
    log_param("pool_size4", hp['pool4'])
    log_param("pool_size5", hp['pool5'])
    log_param("pool_stride1", hp['stridepool1'])
    log_param("pool_stride2", hp['stridepool2'])
    log_param("pool_stride3", hp['stridepool3'])
    log_param("pool_stride4", hp['stridepool4'])
    log_param("pool_stride5", hp['stridepool5'])
    log_param("dense_size1", hp['dense1'])
    log_param("dense_size2", hp['dense2'])
    log_param("learning_rate", hp['lr'])
    log_param("batchsize", hp['batchsize'])
    log_param("patience", hp['patience'])
    log_param("data_augmentation", hp['dataaugment'])

    model = models.CustomVGG16(
        filter1=int(hp['filter1']),
        kernel_size1=int(hp['ksize1']),
        stride1=int(hp['stride1']),
        pool1=int(hp['pool1']),
        stride_pool1=int(hp['stridepool1']),
        filter2=int(hp['filter2']),
        kernel_size2=int(hp['ksize2']),
        stride2=int(hp['stride2']),
        pool2=int(hp['pool2']),
        stride_pool2=int(hp['stridepool2']),
        filter3=int(hp['filter3']),
        kernel_size3=int(hp['ksize3']),
        stride3=int(hp['stride3']),
        pool3=int(hp['pool3']),
        stride_pool3=int(hp['stridepool3']),
        filter4=int(hp['filter4']),
        kernel_size4=int(hp['ksize4']),
        stride4=int(hp['stride4']),
        pool4=int(hp['pool4']),
        stride_pool4=int(hp['stridepool4']),
        filter5=int(hp['filter5']),
        kernel_size5=int(hp['ksize5']),
        stride5=int(hp['stride5']),
        pool5=int(hp['pool5']),
        stride_pool5=int(hp['stridepool5']),
        dense1=int(hp['dense1']),
        dense2=int(hp['dense2']),
        outsize=9
    )

    optimizer = tf.keras.optimizers.Adam(lr=hp['lr'])

    # callback method for checkpointing
    cp_callback = tf.keras.callbacks.ModelCheckpoint(args.ckptpath,
                                                     verbose=0,
                                                     save_weights_only=True,
                                                     save_freq=args.ckptfreq)

    loss = train.fit_loop(train_dataset, valid_dataset, model, optimizer, int(hp['nepoch']), int(hp['batchsize']),
                          augment_policy=hp['dataaugment'], callback=[ModelCallback(int(hp['patience'])), cp_callback])
    test_metrics = train.evaluate_model(model, test_dataset, int(hp['batchsize']))
    log_metric("test_loss", test_metrics[0])
    log_metric("test_acc", test_metrics[1])
    logging.info(f"Test loss / accuracy is {'{:7.2f}'.format(test_metrics[0])} / {'{:7.2f}'.format(test_metrics[1])}")
