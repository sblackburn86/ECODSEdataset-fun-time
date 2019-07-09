import numpy as np
import pytest
import tensorflow as tf

from ecodse_funtime_alpha.cutout import create_cutout_mask
from ecodse_funtime_alpha.cutout import cutout_numpy
from ecodse_funtime_alpha.cutout import cutout_tf

tf.enable_eager_execution()


class TestCutout(object):
    @pytest.fixture(autouse=True)
    def mock_files(self):
        self.img_size = 256
        self.img_channel = 3
        self.img_colorpixel = 128
        self.img_color = 155

        img = np.zeros((self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img[self.img_colorpixel:] = self.img_color
        self.img = img / 255

    def test_createmask(self):
        np.random.seed(0)
        size = 16
        masked, upper, lower = create_cutout_mask(self.img_size, self.img_size, self.img_channel, size)
        assert upper[0] < lower[0]
        assert upper[1] < lower[1]
        assert masked.shape == (self.img_size, self.img_size, self.img_channel)

    def test_cutout(self):
        np.random.seed(0)
        masked_img = cutout_numpy(self.img)
        assert np.any(masked_img != self.img)

    def test_cutout_tf(self):
        np.random.seed(0)
        tf_img = tf.convert_to_tensor(self.img)
        masked_tensor = cutout_tf(tf_img)
        assert masked_tensor.shape == tf_img.shape
