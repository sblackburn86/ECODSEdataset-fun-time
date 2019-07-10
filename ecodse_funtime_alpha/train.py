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
    if scheme in ["cifar10", "imagenet", "svhn"]:
        augment_policy = autoaugment.AugmentationPolicy(dataset=scheme)
        # dataset.map is not done eagerly; using contrib.earg.py_func to get it running
        # will be deprecated in tensorflow 2
        return dataset.map(lambda x: tf.contrib.eager.py_func(augment_policy.call, [x], tf.float64))
    else:
        # no augmentation
        return dataset


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

    def_dataaugmentscheme = None
    argparser.add_argument('-da',
                           '--dataaugment',
                           default=def_dataaugmentscheme,
                           type=str,
                           help=f'Data augmentation scheme based on AutoAugment Implemented options are \n cifar10\n imagenet \n svhn \n(default {def_dataaugmentscheme})')

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
    # model = models.SimpleCNN(args.kernels, args.ksize, 9)
    model = models.CustomVGG16(
        filter1=args.filter1,
        kernel_size1=args.ksize1,
        stride1=args.stride1,
        pool1=args.pool1,
        stride_pool1=args.stridepool1,
        filter2=args.filter2,
        kernel_size2=args.ksize2,
        stride2=args.stride2,
        pool2=args.pool2,
        stride_pool2=args.stridepool2,
        filter3=args.filter3,
        kernel_size3=args.ksize3,
        stride3=args.stride3,
        pool3=args.pool3,
        stride_pool3=args.stridepool3,
        filter4=args.filter4,
        kernel_size4=args.ksize4,
        stride4=args.stride4,
        pool4=args.pool4,
        stride_pool4=args.stridepool4,
        filter5=args.filter5,
        kernel_size5=args.ksize5,
        stride5=args.stride5,
        pool5=args.pool5,
        stride_pool5=args.stridepool5,
        dense1=args.dense1,
        dense2=args.dense2,
        outsize=9
    )
    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
    model = fit_loop(dataset, model, optimizer, args.nepoch, args.batchsize)
