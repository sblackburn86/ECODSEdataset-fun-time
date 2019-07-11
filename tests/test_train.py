import math
import os
import subprocess

from copy import deepcopy
from PIL import Image

import numpy as np
import pytest
import tensorflow as tf

from ecodse_funtime_alpha.train import augment_images
from ecodse_funtime_alpha.train import batch_dataset
from ecodse_funtime_alpha.train import fit_loop
from ecodse_funtime_alpha.train import get_args
from ecodse_funtime_alpha.train import train_loop

tf.enable_eager_execution()


class TestArgparse(object):

    def test_argparsenormal(self):
        fakearg = ['--imagepath=./', '--labelpath=fakedir/name.csv',
                   '--seed=1', '--kernels=10', '--ksize=1',
                   '--lr=0.01', '--nepoch=2', '--batchsize=4'
                   ]
        args = get_args(fakearg)
        assert args.imagepath == './'
        assert args.labelpath == 'fakedir/name.csv'
        assert args.seed == 1
        assert args.kernels == 10
        assert args.ksize == 1
        assert args.lr == 0.01
        assert args.nepoch == 2
        assert args.batchsize == 4

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_lr(self):
        fakearg = ['--lr=a']
        _ = get_args(fakearg)

    @pytest.mark.xfail(raises=SystemExit)
    def test_argparse_seed(self):
        fakearg = ['--seed=a']
        _ = get_args(fakearg)

    def test_argparse_imagepath(self):
        fakearg = ['--imagepath=notavalidpath']
        args = get_args(fakearg)
        assert not os.path.isdir(args.imagepath)

    def test_argparse_labelpath(self):
        fakearg = ['--labelpath=invalid.csv']
        args = get_args(fakearg)
        assert not os.path.exists(args.labelpath)


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
        self.img_ds = tf.data.Dataset.from_tensor_slices(img)

    def test_augmentdataset_noaugment(self):
        scheme = "no-augment"
        init_dataset = self.img_ds
        augmented_dataset = augment_images(self.img_ds, scheme)
        assert all([np.all(x.numpy() == y.numpy()) for x, y in zip(augmented_dataset, init_dataset)])

    def test_augmentdataset_cifar(self):
        scheme = "cifar10"
        init_img = [x.numpy() for x in self.img_ds]
        np.random.seed(2)
        augmented_dataset = augment_images(self.img_ds, scheme)
        assert any([np.any(x.numpy() != y) for x, y in zip(augmented_dataset, init_img)])

    def test_augmentdataset_imagenet(self):
        scheme = "imagenet"
        init_img = [x.numpy() for x in self.img_ds]
        np.random.seed(2)
        augmented_dataset = augment_images(self.img_ds, scheme)
        assert any([np.any(x.numpy() != y) for x, y in zip(augmented_dataset, init_img)])

    def test_augmentdataset_svhn(self):
        scheme = "svhn"
        init_img = [x.numpy() for x in self.img_ds]
        np.random.seed(2)
        augmented_dataset = augment_images(self.img_ds, scheme)
        assert any([np.any(x.numpy() != y) for x, y in zip(augmented_dataset, init_img)])


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
        model = fit_loop(self.dataset, self.dataset, self.model, tf.keras.optimizers.Adam(lr=0.1), nepoch, batch_size)
        after = model.trainable_variables
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


class TestTrainMain(object):
    @pytest.fixture(autouse=True)
    def mock_files(self, tmpdir):
        self.img_size = 256
        self.img_channel = 3
        self.img_colorpixel = 128
        self.img_color = 255

        p = tmpdir.mkdir("train-jpg").join("train_v2.csv")
        csv_text = ["image_name,tags\n"] + [f"train_{x}, a b c d e f g h\n" for x in range(9)] + ["train_9, a b c"]
        p.write_text("".join(csv_text), encoding="utf-8")

        img = np.zeros((self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img[self.img_colorpixel] = self.img_color

        for x in range(10):
            Image.fromarray(img).save(str(tmpdir.join("train-jpg").join(f"train_{x}.jpg")), quality=100)

    def test_main(self, tmpdir):
        imagedir = str(tmpdir.join("train-jpg"))
        labelpath = str(tmpdir.join("train-jpg").join("train_v2.csv"))
        nepoch = 1
        batchsize = 1
        fakearg = ["--imagepath", imagedir, "--labelpath", labelpath, "--nepoch", str(nepoch), "--batchsize", str(batchsize)]
        assert subprocess.call(["python", "ecodse_funtime_alpha/train.py", *fakearg]) == 0
