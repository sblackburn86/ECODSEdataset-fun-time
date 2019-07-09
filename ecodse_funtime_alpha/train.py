import argparse
import math
import sys

import tensorflow as tf

import ecodse_funtime_alpha.autoaugment as autoaugment
import ecodse_funtime_alpha.data as data
import ecodse_funtime_alpha.models as models


tf.enable_eager_execution()


def batch_dataset(dataset, nepoch, batchsize):
    """
    Shuffle, repeat and split a dataset into mini-batches.
    To avoid running out of data, the original dataset is repeated nepoch times.
    All mini-batches have the same size, except the last one (remainder) in each epoch.
    Data are shuffled randomly in 1 epoch (each data element occurs once in 1 epoch).

    Parameters
    ----------
    dataset : tf dataset
        initial dataset, unshuffled, not repeated and not split into mini-batches
    nepoch : int
        number of epochs that will be used in the training
    batchsize : int
        size of the mini-batches. Should be lower than the number of elements in the dataset

    Returns
    -------
    tf dataset
        shuffled dataset split into mini-batches
    """
    # shuffling the dataset before mini-batches to shuffle elements and not mini-batches
    dataset = dataset.shuffle(buffer_size=10 * batchsize)
    # split into mini-batches
    dataset = dataset.batch(batchsize)
    # repeat for multiple epochs; the earlier shuffle is different for each epoch
    dataset = dataset.repeat(nepoch)
    return dataset


def augment_images(dataset, scheme="cifar10"):
    """
    Augment images according to a given scheme

    Parameters
    ----------
    dataset : tf dataset
        initial dataset, unshuffled, not repeated and not split into mini-batches
    scheme : str, optional
        data augmentation method, by default "auto-augment-cifar10"
            cifar10: use method described in https://arxiv.org/abs/1805.09501 for CIFAR-10
            imagenet: use auto-augment method for ImageNet
            svhn: use auto-augment method for SVHN
            no-augment: do not use any image augmentation
    """
    if scheme == "cifar10":
        augment_policy = autoaugment.CIFAR10_policy()
    elif scheme == "imagenet":
        augment_policy = autoaugment.ImageNet_policy()
    elif scheme == "svhn":
        augment_policy = autoaugment.SVHN_policy()
    else:
        # no augmentation
        return dataset
    # dataset.map is not done eagerly; using contrib.earg.py_func to get it running
    # will be deprecated in tensorflow 2
    return dataset.map(lambda x: tf.contrib.eager.py_func(augment_policy.call, [x], tf.float64))


def train_loop(dataset, model, optimizer, nepoch, batchsize):
    """
    Training loop feeding the mini-batches in the dataset in the model one at a time.
    Gradient is applied manually on the model using the optimizer.

    Parameters
    ----------
    dataset : tf dataset
        original dataset (unshuffled, not-split into mini-batches)
    model : tf.keras.Model
        an initialized model working in eager execution mode
    optimizer : tf.train.Optimizer
        tensorflow optimizer (e.g. `tf.train.AdamOptimizer()`) to train the model
    nepoch : int
        number of epochs to train the model
    batchsize : int
        size of the mini-batches

    Returns
    -------
    tf.keras.Model
        model after training
    """
    dataset = batch_dataset(dataset, nepoch, batchsize)
    for x, y in dataset:
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.cast(y, tf.float32), logits=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return model


def fit_loop(dataset, model, optimizer, nepoch, batchsize, augment_policy="no-augment"):
    """
    Training loop fitting the model using the keras .fit() method

    Parameters
    ----------
    dataset : tf dataset
        original dataset (unshuffled, not-split into mini-batches)
    model : tf.keras.Model
        an initialized model working in eager execution mode
    optimizer : tf.keras.optimizers
        tf.keras optimizer (e.g. `tf.keras.optimizers.Adam()`) to train the model
    nepoch : int
        number of epochs to train the model
    batchsize : int
        size of the mini-batches
    augment_policy: str, optional
        image augmentation policy to apply, by default no-augment

    Returns
    -------
    tf.keras.Model
        model after training
    """
    # number of steps in an epoch is len(dataset) / batchsize (math.ceil for the remainder)
    nstep = math.ceil(tf.data.experimental.cardinality(dataset).numpy() / batchsize)
    dataset = augment_images(dataset, scheme=augment_policy)
    dataset = batch_dataset(dataset, nepoch, batchsize)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
    model.fit(dataset, epochs=nepoch, steps_per_epoch=nstep)
    return model


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
    def_impath = '../../rainforest/fixed-train-jpg/'
    argparser.add_argument('--imagepath',
                           default=def_impath,
                           help=f'path to image directory (default {def_impath})')
    def_labelpath = '../../rainforest/train_v3.csv'
    argparser.add_argument('--labelpath',
                           default=def_labelpath,
                           help=f'path to csv file for labels (defautlt {def_labelpath})')
    def_seed = -1
    argparser.add_argument('-s',
                           '--seed',
                           default=def_seed,
                           type=int,
                           help=f'Set random seed to this number (default {def_seed})')
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
    def_batch = 4
    argparser.add_argument('-b',
                           '--batchsize',
                           default=def_batch,
                           type=int,
                           help=f'batch size (default {def_batch})')
    args = argparser.parse_args(args)
    return args


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    tf.random.set_random_seed(args.seed)
    dataset = data.get_dataset(args.imagepath, args.labelpath)
    # model = models.TestMLP(10, 9)
    model = models.SimpleCNN(args.kernels, args.ksize, 9)
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    model = fit_loop(dataset, model, optimizer, args.nepoch, args.batchsize)
