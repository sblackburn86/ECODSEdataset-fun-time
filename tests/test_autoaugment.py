import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from ecodse_funtime_alpha.autoaugment import autocontrast
from ecodse_funtime_alpha.autoaugment import brightness
from ecodse_funtime_alpha.autoaugment import CIFAR10_policy
from ecodse_funtime_alpha.autoaugment import color
from ecodse_funtime_alpha.autoaugment import contrast
from ecodse_funtime_alpha.autoaugment import equalize
from ecodse_funtime_alpha.autoaugment import ImageNet_policy
from ecodse_funtime_alpha.autoaugment import invert
from ecodse_funtime_alpha.autoaugment import pil_unwrap
from ecodse_funtime_alpha.autoaugment import pil_wrap
from ecodse_funtime_alpha.autoaugment import posterize
from ecodse_funtime_alpha.autoaugment import rotate
from ecodse_funtime_alpha.autoaugment import sharpness
from ecodse_funtime_alpha.autoaugment import shear
from ecodse_funtime_alpha.autoaugment import solarize
from ecodse_funtime_alpha.autoaugment import SVHN_policy
from ecodse_funtime_alpha.autoaugment import translate

tf.enable_eager_execution()


class TestAutoAugment(object):
    @pytest.fixture(autouse=True)
    def mock_files(self):
        self.img_size = 256
        self.img_channel = 3
        self.img_colorpixel = 128
        self.img_color = 155

        img = np.zeros((self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img[self.img_colorpixel:] = self.img_color
        self.tf_img = tf.convert_to_tensor(img, dtype=tf.float32) / 255
        self.pil_img = Image.fromarray(img)

        img2 = np.zeros((self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img2[:, self.img_colorpixel:] = self.img_color
        self.tf_img2 = tf.convert_to_tensor(img2, dtype=tf.float32) / 255

    def test_wrap(self):
        pil_img = pil_wrap(self.tf_img)
        assert np.array(pil_img.getdata()).shape == (self.img_size ** 2, self.img_channel)
        assert np.all(np.array(pil_img.getdata()).reshape(self.img_size, self.img_size, self.img_channel)[self.img_colorpixel] == self.img_color)

    def test_unwrap(self):
        tf_img = pil_unwrap(self.pil_img)
        assert tf_img.shape == (256, 256, 3)
        assert np.all(tf_img[self.img_colorpixel].numpy() == self.img_color / 255)

    def test_autocontrast(self):
        ac_img = autocontrast(self.tf_img)
        assert ac_img.shape == (self.img_size, self.img_size, self.img_channel)

    def test_brightness(self):
        magnitude = 5
        bright_img = brightness(self.tf_img, magnitude)
        assert bright_img.shape == (self.img_size, self.img_size, self.img_channel)

    def test_color(self):
        magnitude = 5
        color_img = color(self.tf_img, magnitude)
        assert color_img.shape == (self.img_size, self.img_size, self.img_channel)

    def test_contrast(self):
        magnitude = 5
        contrast_img = contrast(self.tf_img, magnitude)
        assert contrast_img.shape == (self.img_size, self.img_size, self.img_channel)

    def test_equalize(self):
        eq_img = equalize(self.tf_img)
        assert eq_img.shape == (self.img_size, self.img_size, self.img_channel)

    def test_invert(self):
        invert_img = invert(self.tf_img)
        assert np.all(abs(invert_img[self.img_colorpixel].numpy() - (1 - self.img_color / 255)) < 1e-3)

    def test_posterize(self):
        magnitude = 5
        post_img = posterize(self.tf_img, magnitude)
        assert post_img.shape == (self.img_size, self.img_size, self.img_channel)

    def test_rotate(self):
        magnitude = 5
        # np.random.random() > 0.5
        np.random.seed(0)
        rotate_img = rotate(self.tf_img, magnitude)
        assert np.all(rotate_img[self.img_colorpixel, 0].numpy() == self.img_color / 255)
        assert np.all(rotate_img[self.img_colorpixel, -1].numpy() == 0)
        # np.random.random() < 0.5
        np.random.seed(1)
        rotate_img = rotate(self.tf_img, magnitude)
        assert np.all(rotate_img[self.img_colorpixel, 0].numpy() == 0)
        assert np.all(rotate_img[self.img_colorpixel, -1].numpy() == self.img_color / 255)

    def test_sharpness(self):
        magnitude = 5
        sharp_img = sharpness(self.tf_img, magnitude)
        assert sharp_img.shape == (self.img_size, self.img_size, self.img_channel)

    def test_shearX(self):
        shear_magnitude = 4
        # np.random.random() > 0.5
        np.random.seed(0)
        shear_img = shear(self.tf_img, shear_magnitude, axis="x")
        assert np.all(shear_img[self.img_colorpixel, 0].numpy() == 0)
        assert np.all(shear_img[self.img_colorpixel, -1].numpy() == self.img_color / 255)
        # np.random.random() < 0.5
        np.random.seed(1)
        shear_img = shear(self.tf_img, shear_magnitude, axis="x")
        assert np.all(shear_img[self.img_colorpixel, 0].numpy() == self.img_color / 255)
        assert np.all(shear_img[self.img_colorpixel, -1].numpy() == 0)

    def test_shearY(self):
        shear_magnitude = 4
        # np.random.random() > 0.5
        np.random.seed(0)
        shear_img = shear(self.tf_img2, shear_magnitude, axis="y")
        assert np.all(shear_img[0, self.img_colorpixel].numpy() == 0)
        assert np.all(shear_img[-1, self.img_colorpixel].numpy() == self.img_color / 255)
        # np.random.random() < 0.5
        np.random.seed(1)
        shear_img = shear(self.tf_img2, shear_magnitude, axis="y")
        assert np.all(shear_img[0, self.img_colorpixel].numpy() == self.img_color / 255)
        assert np.all(shear_img[-1, self.img_colorpixel].numpy() == 0)

    def test_solarize(self):
        magnitude = 5
        sol_img = solarize(self.tf_img, magnitude)
        assert sol_img.shape == (self.img_size, self.img_size, self.img_channel)

    def test_translateX(self):
        magnitude = 5
        # np.random.random() > 0.5
        np.random.seed(0)
        translate_img = translate(self.tf_img2, magnitude, axis="x")
        assert np.all(translate_img[:, magnitude:self.img_colorpixel + magnitude].numpy() == 0)
        assert np.all(translate_img[:, self.img_colorpixel + magnitude:].numpy() == self.img_color / 255)
        # np.random.random() < 0.5
        np.random.seed(1)
        translate_img = translate(self.tf_img2, magnitude, axis="x")
        assert np.all(translate_img[:, :self.img_colorpixel - magnitude].numpy() == 0)
        assert np.all(translate_img[:, self.img_colorpixel - magnitude:-magnitude].numpy() == self.img_color / 255)
        assert np.all(translate_img[:, -magnitude:].numpy() == 0)

    def test_translateY(self):
        magnitude = 5
        # np.random.random() > 0.5
        np.random.seed(0)
        translate_img = translate(self.tf_img, magnitude, axis="y")
        assert np.all(translate_img[magnitude:self.img_colorpixel + magnitude].numpy() == 0)
        assert np.all(translate_img[self.img_colorpixel + magnitude:].numpy() == self.img_color / 255)
        # np.random.random() < 0.5
        np.random.seed(1)
        translate_img = translate(self.tf_img, magnitude, axis="y")
        assert np.all(translate_img[:self.img_colorpixel - magnitude].numpy() == 0)
        assert np.all(translate_img[self.img_colorpixel - magnitude:-magnitude].numpy() == self.img_color / 255)
        assert np.all(translate_img[-magnitude:].numpy() == 0)

    def test_cifar10_applytransform(self):
        augment_policy = CIFAR10_policy()
        magnitude = 8
        autocontrast_img = augment_policy.apply_transform(self.tf_img, autocontrast, 1, magnitude)
        assert np.all(autocontrast_img.numpy() == autocontrast(self.tf_img, magnitude).numpy())
        id_img = augment_policy.apply_transform(self.tf_img, autocontrast, 0, magnitude)
        assert np.all(id_img == self.tf_img)
        np.random.seed(0)
        shear_img = augment_policy.apply_transform(self.tf_img, shear, 1, magnitude, "x")
        np.random.seed(0)
        assert np.all(abs(shear_img.numpy() - shear(self.tf_img, magnitude, "x")) < 1e-3)

    def test_cifar10_call(self):
        augment_policy = CIFAR10_policy()
        np.random.seed(0)
        aug_img = augment_policy.call(self.tf_img)
        np.random.seed(0)
        transformations = augment_policy.subpolicies[np.random.randint(len(augment_policy.subpolicies))]
        manual_img = self.tf_img
        for t in transformations:
            if np.random.random() < t[1]:
                if len(t) == 3:
                    manual_img = t[0](manual_img, t[2])
                else:
                    manual_img = t[0](manual_img, t[2], t[3])
        assert np.all(abs(aug_img - manual_img) < 1e-3)

    def test_svhn_applytransform(self):
        augment_policy = SVHN_policy()
        magnitude = 8
        autocontrast_img = augment_policy.apply_transform(self.tf_img, autocontrast, 1, magnitude)
        assert np.all(autocontrast_img.numpy() == autocontrast(self.tf_img, magnitude).numpy())
        id_img = augment_policy.apply_transform(self.tf_img, autocontrast, 0, magnitude)
        assert np.all(id_img == self.tf_img)
        np.random.seed(0)
        shear_img = augment_policy.apply_transform(self.tf_img, shear, 1, magnitude, "x")
        np.random.seed(0)
        assert np.all(abs(shear_img.numpy() - shear(self.tf_img, magnitude, "x")) < 1e-3)

    def test_svhn_call(self):
        augment_policy = SVHN_policy()
        np.random.seed(0)
        aug_img = augment_policy.call(self.tf_img)
        np.random.seed(0)
        transformations = augment_policy.subpolicies[np.random.randint(len(augment_policy.subpolicies))]
        manual_img = self.tf_img
        for t in transformations:
            if np.random.random() < t[1]:
                if len(t) == 3:
                    manual_img = t[0](manual_img, t[2])
                else:
                    manual_img = t[0](manual_img, t[2], t[3])
        assert np.all(abs(aug_img - manual_img) < 1e-3)

    def test_imagenet_applytransform(self):
        augment_policy = ImageNet_policy()
        magnitude = 8
        autocontrast_img = augment_policy.apply_transform(self.tf_img, autocontrast, 1, magnitude)
        assert np.all(autocontrast_img.numpy() == autocontrast(self.tf_img, magnitude).numpy())
        id_img = augment_policy.apply_transform(self.tf_img, autocontrast, 0, magnitude)
        assert np.all(id_img == self.tf_img)
        np.random.seed(0)
        shear_img = augment_policy.apply_transform(self.tf_img, shear, 1, magnitude, "x")
        np.random.seed(0)
        assert np.all(abs(shear_img.numpy() - shear(self.tf_img, magnitude, "x")) < 1e-3)

    def test_imagenet_call(self):
        augment_policy = ImageNet_policy()
        np.random.seed(0)
        aug_img = augment_policy.call(self.tf_img)
        np.random.seed(0)
        transformations = augment_policy.subpolicies[np.random.randint(len(augment_policy.subpolicies))]
        manual_img = self.tf_img
        for t in transformations:
            if np.random.random() < t[1]:
                if len(t) == 3:
                    manual_img = t[0](manual_img, t[2])
                else:
                    manual_img = t[0](manual_img, t[2], t[3])
        assert np.all(abs(aug_img - manual_img) < 1e-3)
