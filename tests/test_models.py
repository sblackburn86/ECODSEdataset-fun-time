import numpy as np
import pytest
import tensorflow as tf

from ecodse_funtime_alpha.models import FuntimeResnet50, PretrainedVGG16, SimpleCNN, TestMLP


class TestTestMLP(object):
    @pytest.fixture(autouse=True)
    def create_model(self):
        self.batchsize = 2
        self.insize = 256 * 256 * 3
        self.hiddensize = 5
        self.outsize = 9
        self.testmodel = TestMLP(self.hiddensize, self.outsize)
        self.testinput = tf.zeros([self.batchsize, self.insize])

    def test_outshape(self):
        testout = self.testmodel(self.testinput)
        assert np.array_equal(tf.shape(testout).numpy(), [self.batchsize, self.outsize])

    def test_numparam(self):
        _ = self.testmodel(self.testinput)
        nparam = np.sum([np.product([xi for xi in x.get_shape()]) for x in self.testmodel.trainable_variables])
        assert nparam == (self.insize + 1) * self.hiddensize + (self.hiddensize + 1) * self.outsize


class TestSimpleCNN(object):
    @pytest.fixture(autouse=True)
    def create_model(self):
        self.batchsize = 2
        self.img_size = 256
        self.img_channel = 3
        self.kernels = 5
        self.kernel_size = 4
        self.outsize = 9
        self.testmodel = SimpleCNN(self.kernels, self.kernel_size, self.outsize)
        self.testinput = tf.zeros([self.batchsize, self.img_size, self.img_size, self.img_channel])

    def test_outshape(self):
        testout = self.testmodel(self.testinput)
        assert np.array_equal(tf.shape(testout).numpy(), [self.batchsize, self.outsize])

    def test_numparam(self):
        _ = self.testmodel(self.testinput)
        nparam = np.sum([np.product([xi for xi in x.get_shape()]) for x in self.testmodel.trainable_variables])
        assert nparam == (self.kernel_size * self.kernel_size * self.img_channel + 1) * self.kernels + \
                         (self.kernels * ((self.img_size - (self.kernel_size - 1)) // 2) ** 2 + 1) * self.outsize


class TestFuntimeResnet50(object):
    @pytest.fixture(autouse=True)
    def create_model(self):
        self.batchsize = 2
        self.img_size = 256
        self.img_channel = 3
        self.outsize = 9
        self.frozenmodel = FuntimeResnet50(self.outsize, train_resnet=False)
        self.notfrozenmodel = FuntimeResnet50(self.outsize, train_resnet=True)
        self.testinput = tf.zeros([self.batchsize, self.img_size, self.img_size, self.img_channel])

    def test_outshape(self):
        out_frozen = self.frozenmodel(self.testinput)
        out_notfrozen = self.notfrozenmodel(self.testinput)
        assert np.array_equal(tf.shape(out_frozen).numpy(), [self.batchsize, self.outsize])
        assert np.array_equal(tf.shape(out_notfrozen).numpy(), [self.batchsize, self.outsize])

    def test_numparam(self):
        _ = self.frozenmodel(self.testinput)
        _ = self.notfrozenmodel(self.testinput)
        nparam_frozen = np.sum([np.product([xi for xi in x.get_shape()]) for x in self.frozenmodel.trainable_variables])
        nparam_notfrozen = np.sum([np.product([xi for xi in x.get_shape()]) for x in self.notfrozenmodel.trainable_variables])
        assert nparam_frozen == (2 * self.img_size ** 2 + 1) * self.outsize
        assert nparam_notfrozen == 23534592 + (2 * self.img_size ** 2 + 1) * self.outsize


class TestPretrainedVGG16(object):
    @pytest.fixture(autouse=True)
    def create_data(self):
        self.batchsize = 2
        self.img_size = 256
        self.img_channel = 3
        self.outsize = 9
        self.testinput = tf.zeros([self.batchsize, self.img_size, self.img_size, self.img_channel])

    def test_outshape_wopooling(self):
        model_frozen = PretrainedVGG16(self.outsize, train_vgg=False, pooling=None)
        model_notfrozen = PretrainedVGG16(self.outsize, train_vgg=True, pooling=None)
        out_frozen = model_frozen(self.testinput)
        out_notfrozen = model_notfrozen(self.testinput)
        assert np.array_equal(tf.shape(out_frozen).numpy(), [self.batchsize, self.outsize])
        assert np.array_equal(tf.shape(out_notfrozen).numpy(), [self.batchsize, self.outsize])

    def test_poolsize(self):
        model_avg = PretrainedVGG16(self.outsize, train_vgg=False, pooling="avg")
        model_max = PretrainedVGG16(self.outsize, train_vgg=False, pooling="max")
        out_avg = model_avg(self.testinput)
        out_max = model_max(self.testinput)
        assert np.array_equal(tf.shape(out_avg).numpy(), [self.batchsize, self.outsize])
        assert np.array_equal(tf.shape(out_max).numpy(), [self.batchsize, self.outsize])

    def test_poolassert(self):
        with pytest.raises(AssertionError):
            _ = PretrainedVGG16(self.outsize, train_vgg=False, pooling=123)
