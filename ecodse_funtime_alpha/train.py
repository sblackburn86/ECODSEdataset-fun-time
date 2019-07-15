import math

import tensorflow as tf

import ecodse_funtime_alpha.autoaugment as autoaugment

tf.compat.v1.enable_eager_execution()


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
        return dataset.map(lambda x, y: (tf.contrib.eager.py_func(augment_policy.call, [x], tf.float64), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
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


def fit_loop(dataset, valid_dataset, model, optimizer, nepoch, batchsize, augment_policy=None, callback=None):
    """
    Training loop fitting the model using the keras .fit() method

    Parameters
    ----------
    dataset : tf dataset
        training dataset (unshuffled, not-split into mini-batches)
    valid_dataset : tf dataset
        validation dataset
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
    callback : tf.keras.callbacks.Callback
        callback class for logging in the fit loop

    Returns
    -------
    tensorflow.python.keras.callbacks.History
        training history
    """
    # number of steps in an epoch is len(dataset) / batchsize (math.ceil for the remainder)
    nstep = math.ceil(tf.data.experimental.cardinality(dataset).numpy() / batchsize)
    dataset = augment_images(dataset, scheme=augment_policy)
    dataset = batch_dataset(dataset, nepoch, batchsize)
    valid_dataset = valid_dataset.batch(batchsize)
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=["accuracy"])
    history = model.fit(dataset, epochs=nepoch, steps_per_epoch=nstep, validation_data=valid_dataset,
                        verbose=0, callbacks=callback)
    return history


def evaluate_model(model, test_dataset, batchsize=4):
    """evaluate the model on the test dataset

    Parameters
    ----------
    model : tf.keras.Model
        trained model to evaluate
    test_dataset : tf dataset
        test dataset
    batchsize : int, optional
        size of mini-batches, by default 4

    Returns
    -------
    list of float
        resulting metrics (loss & accuracy)
    """
    test_dataset = test_dataset.batch(batchsize)
    results = model.evaluate(test_dataset)
    return results
