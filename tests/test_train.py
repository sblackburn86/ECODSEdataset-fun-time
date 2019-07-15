import math

import numpy as np
import pytest
import tensorflow as tf

from copy import deepcopy
from ecodse_funtime_alpha.train import augment_images
from ecodse_funtime_alpha.train import batch_dataset
from ecodse_funtime_alpha.train import fit_loop
from ecodse_funtime_alpha.train import train_loop

tf.compat.v1.enable_eager_execution()


class TestBatchDataset(object):
    @pytest.fixture(autouse=True)
    def mock_file(self):
        self.nimage = 10
        self.out_size = 9
        self.image_size = 256 * 256 * 3
        img_ds = tf.data.Dataset.from_tensor_slices(tf.zeros([self.nimage, self.image_size]))
        label_ds = tf.data.Dataset.from_tensor_slices(tf.zeros([self.nimage, self.out_size]))
        self.dataset = tf.data.Dataset.zip((img_ds, label_ds))
        self.img_ds = img_ds

    def test_dataset(self):
        batch_size = 4
        nepoch = 3
        dataset = batch_dataset(self.dataset, nepoch, batch_size)
        # get sizes of mini-batches in one epoch
        size_of_batch = [batch_size] * (self.nimage // batch_size)
        # add remainder if number of examples is not a multiple of batchsize
        size_of_batch += [self.nimage % batch_size] if self.nimage % batch_size != 0 else []
        # multiply by number of epochs
        size_of_batch = [*size_of_batch] * nepoch
        assert [x[0].shape[0].value for x in dataset] == size_of_batch

    def test_datasetsize(self):
        batch_size = 1
        nepoch = 2
        dataset = batch_dataset(self.dataset, nepoch, batch_size)
        assert tf.data.experimental.cardinality(dataset).numpy() == math.ceil(self.nimage / batch_size) * nepoch


class TestAugmentDataset(object):
    @pytest.fixture(autouse=True)
    def mock_files(self):
        self.img_size = 256
        self.img_channel = 3
        self.img_colorpixel = 128
        self.img_color = 155
        img = np.zeros((1, self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img[0, self.img_colorpixel:] = self.img_color
        img_ds = tf.data.Dataset.from_tensor_slices(img)
        label_ds = tf.data.Dataset.from_tensor_slices(tf.zeros([1, 1]))
        self.dataset = tf.data.Dataset.zip((img_ds, label_ds))

    def test_augmentdataset_noaugment(self):
        scheme = "no-augment"
        init_dataset = self.dataset
        augmented_dataset = augment_images(self.dataset, scheme)
        assert all([np.all(x[0].numpy() == y[0].numpy()) for x, y in zip(augmented_dataset, init_dataset)])

    def test_augmentdataset_cifar(self):
        scheme = "cifar10"
        init_img = [x.numpy() for x, _ in self.dataset]
        np.random.seed(2)
        augmented_dataset = augment_images(self.dataset, scheme)
        assert any([np.any(x.numpy() != y) for (x, _), y in zip(augmented_dataset, init_img)])

    def test_augmentdataset_imagenet(self):
        scheme = "imagenet"
        init_img = [x.numpy() for x, _ in self.dataset]
        np.random.seed(2)
        augmented_dataset = augment_images(self.dataset, scheme)
        assert any([np.any(x.numpy() != y) for (x, _), y in zip(augmented_dataset, init_img)])

    def test_augmentdataset_svhn(self):
        scheme = "svhn"
        init_img = [x.numpy() for x, _ in self.dataset]
        np.random.seed(2)
        augmented_dataset = augment_images(self.dataset, scheme)
        assert any([np.any(x.numpy() != y) for (x, _), y in zip(augmented_dataset, init_img)])


class TestFitLoop(object):
    @pytest.fixture(autouse=True)
    def mock_file(self):
        self.nimage = 1
        self.in_size = 256 * 256 * 3
        self.out_size = 9
        img_ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform([self.nimage, self.in_size]))
        label_ds = tf.data.Dataset.from_tensor_slices(tf.random.uniform([self.nimage, self.out_size]))
        self.dataset = tf.data.Dataset.zip((img_ds, label_ds))
        self.model = tf.keras.Sequential([tf.keras.layers.Dense(self.out_size, input_shape=(self.in_size,))])

    def test_fitvarchanged(self):
        nepoch = 1
        batch_size = 1
        before = deepcopy(self.model.trainable_variables)
        _ = fit_loop(self.dataset, self.dataset, self.model, tf.keras.optimizers.Adam(lr=0.1), nepoch, batch_size)
        after = self.model.trainable_variables
        for b, a in zip(before, after):
            # make sure something changed
            assert (b.numpy() != a.numpy()).any()

    def test_trainvarchanged(self):
        nepoch = 1
        batch_size = 1
        before = deepcopy(self.model.trainable_variables)
        model = train_loop(self.dataset, self.model, tf.train.AdamOptimizer(learning_rate=0.1), nepoch, batch_size)
        after = model.trainable_variables
        for b, a in zip(before, after):
            # make sure something changed
            assert (b.numpy() != a.numpy()).any()
