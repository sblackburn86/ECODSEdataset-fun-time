import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from ecodse_funtime_alpha.autoaugment import AugmentationPolicy
from ecodse_funtime_alpha.autoaugment import autocontrast
from ecodse_funtime_alpha.autoaugment import brightness
from ecodse_funtime_alpha.autoaugment import color
from ecodse_funtime_alpha.autoaugment import contrast
from ecodse_funtime_alpha.autoaugment import equalize
from ecodse_funtime_alpha.autoaugment import flip
from ecodse_funtime_alpha.autoaugment import invert
from ecodse_funtime_alpha.autoaugment import posterize
from ecodse_funtime_alpha.autoaugment import rotate
from ecodse_funtime_alpha.autoaugment import sharpness
from ecodse_funtime_alpha.autoaugment import shear
from ecodse_funtime_alpha.autoaugment import solarize
from ecodse_funtime_alpha.autoaugment import translate
from ecodse_funtime_alpha.autoaugment import zero_pad_and_crop
from ecodse_funtime_alpha.cutout import cutout_numpy

tf.compat.v1.enable_eager_execution()


class TestAutoAugment(object):
    @pytest.fixture(autouse=True)
    def mock_files(self):
        self.img_size = 256
        self.img_channel = 3
        self.img_colorpixel = 128
        self.img_color = 155
        self.img_shape = (self.img_size, self.img_size, self.img_channel)

        img = np.zeros((self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img[self.img_colorpixel:] = self.img_color
        self.np_img = img
        self.tf_img = tf.convert_to_tensor(img, dtype=tf.float32) / 255
        self.pil_img = Image.fromarray(img)

        img2 = np.zeros((self.img_size, self.img_size, self.img_channel), dtype=np.uint8)
        img2[:, self.img_colorpixel:] = self.img_color
        self.pil_img2 = Image.fromarray(img2)
        self.tf_img2 = tf.convert_to_tensor(img2, dtype=tf.float32) / 255

    def test_autocontrast(self):
        ac_img = autocontrast(self.pil_img)
        assert ac_img.size == (self.img_size, self.img_size)
        assert ac_img.mode == "RGB"

    def test_brightness(self):
        magnitude = 5
        bright_img = brightness(self.pil_img, magnitude)
        assert bright_img.size == (self.img_size, self.img_size)
        assert bright_img.mode == "RGB"

    def test_color(self):
        magnitude = 5
        color_img = color(self.pil_img, magnitude)
        assert color_img.size == (self.img_size, self.img_size)
        assert color_img.mode == "RGB"

    def test_contrast(self):
        magnitude = 5
        contrast_img = contrast(self.pil_img, magnitude)
        assert contrast_img.size == (self.img_size, self.img_size)
        assert contrast_img.mode == "RGB"

    def test_equalize(self):
        eq_img = equalize(self.pil_img)
        assert eq_img.size == (self.img_size, self.img_size)
        assert eq_img.mode == "RGB"

    def test_flip(self):
        np.random.seed(0)
        flip_img = flip(self.np_img)
        assert flip_img.shape == self.np_img.shape

    def test_invert(self):
        invert_img = invert(self.pil_img)
        assert invert_img.size == (self.img_size, self.img_size)
        assert invert_img.mode == "RGB"

    def test_posterize(self):
        magnitude = 5
        post_img = posterize(self.pil_img, magnitude)
        assert post_img.size == (self.img_size, self.img_size)
        assert post_img.mode == "RGB"

    def test_rotate(self):
        magnitude = 5
        # np.random.random() > 0.5
        np.random.seed(0)
        rotate_img = rotate(self.pil_img, magnitude)
        assert rotate_img.size == (self.img_size, self.img_size)
        assert rotate_img.mode == "RGB"
        # np.random.random() < 0.5
        np.random.seed(1)
        rotate_img = rotate(self.pil_img, magnitude)
        assert rotate_img.size == (self.img_size, self.img_size)
        assert rotate_img.mode == "RGB"

    def test_sharpness(self):
        magnitude = 5
        sharp_img = sharpness(self.pil_img, magnitude)
        assert sharp_img.size == (self.img_size, self.img_size)
        assert sharp_img.mode == "RGB"

    def test_shearX(self):
        shear_magnitude = 4
        # np.random.random() > 0.5
        np.random.seed(0)
        shear_img = shear(self.pil_img, shear_magnitude, axis="x")
        shear_img = np.array(shear_img.getdata()).reshape(self.img_shape)
        assert np.all(shear_img[self.img_colorpixel, 0] == 0)
        assert np.all(shear_img[self.img_colorpixel, -1] == self.img_color)
        # np.random.random() < 0.5
        np.random.seed(1)
        shear_img = shear(self.pil_img, shear_magnitude, axis="x")
        shear_img = np.array(shear_img.getdata()).reshape(self.img_shape)
        assert np.all(shear_img[self.img_colorpixel, 0] == self.img_color)
        assert np.all(shear_img[self.img_colorpixel, -1] == 0)

    def test_shearY(self):
        shear_magnitude = 4
        # np.random.random() > 0.5
        np.random.seed(0)
        shear_img = shear(self.pil_img2, shear_magnitude, axis="y")
        shear_img = np.array(shear_img.getdata()).reshape(self.img_shape)
        assert np.all(shear_img[0, self.img_colorpixel] == 0)
        assert np.all(shear_img[-1, self.img_colorpixel] == self.img_color)
        # np.random.random() < 0.5
        np.random.seed(1)
        shear_img = shear(self.pil_img2, shear_magnitude, axis="y")
        shear_img = np.array(shear_img.getdata()).reshape(self.img_shape)
        assert np.all(shear_img[0, self.img_colorpixel] == self.img_color)
        assert np.all(shear_img[-1, self.img_colorpixel] == 0)

    def test_solarize(self):
        magnitude = 5
        sol_img = solarize(self.pil_img, magnitude)
        assert sol_img.size == (self.img_size, self.img_size)
        assert sol_img.mode == "RGB"

    def test_translateX(self):
        magnitude = 5
        # np.random.random() > 0.5
        np.random.seed(0)
        translate_img = translate(self.pil_img2, magnitude, axis="x")
        translate_img = np.array(translate_img.getdata()).reshape(self.img_shape)
        assert np.all(translate_img[:, magnitude:self.img_colorpixel + magnitude] == 0)
        assert np.all(translate_img[:, self.img_colorpixel + magnitude:] == self.img_color)
        # np.random.random() < 0.5
        np.random.seed(1)
        translate_img = translate(self.pil_img2, magnitude, axis="x")
        translate_img = np.array(translate_img.getdata()).reshape(self.img_shape)
        assert np.all(translate_img[:, :self.img_colorpixel - magnitude] == 0)
        assert np.all(translate_img[:, self.img_colorpixel - magnitude:-magnitude] == self.img_color)
        assert np.all(translate_img[:, -magnitude:] == 0)

    def test_translateY(self):
        magnitude = 5
        # np.random.random() > 0.5
        np.random.seed(0)
        translate_img = translate(self.pil_img, magnitude, axis="y")
        translate_img = np.array(translate_img.getdata()).reshape(self.img_shape)
        assert np.all(translate_img[magnitude:self.img_colorpixel + magnitude] == 0)
        assert np.all(translate_img[self.img_colorpixel + magnitude:] == self.img_color)
        # np.random.random() < 0.5
        np.random.seed(1)
        translate_img = translate(self.pil_img, magnitude, axis="y")
        translate_img = np.array(translate_img.getdata()).reshape(self.img_shape)
        assert np.all(translate_img[:self.img_colorpixel - magnitude] == 0)
        assert np.all(translate_img[self.img_colorpixel - magnitude:-magnitude] == self.img_color)
        assert np.all(translate_img[-magnitude:] == 0)

    def test_zeropadandcrop(self):
        amount = 4
        transformed_img = zero_pad_and_crop(self.np_img, amount)
        assert np.any(transformed_img != self.np_img)

    def test_cifar10_applytransform(self):
        augment_policy = AugmentationPolicy(dataset="cifar10")
        magnitude = 8
        autocontrast_img = augment_policy.apply_transform(self.pil_img, autocontrast, 1, magnitude)
        autocontrast_img = np.array(autocontrast_img.getdata()).reshape(self.img_shape)
        autocontrast_manual = np.array(autocontrast(self.pil_img, magnitude).getdata()).reshape(self.img_shape)
        assert np.all(autocontrast_img == autocontrast_manual)
        id_img = augment_policy.apply_transform(self.pil_img, autocontrast, 0, magnitude)
        id_img = np.array(id_img.getdata()).reshape(self.img_shape)
        assert np.all(id_img == self.np_img)
        np.random.seed(0)
        shear_img = augment_policy.apply_transform(self.pil_img, shear, 1, magnitude, "x")
        shear_img = np.array(shear_img.getdata()).reshape(self.img_shape)
        np.random.seed(0)
        shear_manual = np.array(shear(self.pil_img, magnitude, "x").getdata()).reshape(self.img_shape)
        assert np.all(abs(shear_img - shear_manual) < 1e-3)

    def test_cifar10_call(self):
        augment_policy = AugmentationPolicy(dataset="cifar10")
        np.random.seed(1)
        aug_img = augment_policy.call(self.tf_img, test=True)
        np.random.seed(1)
        manual_img = flip(self.np_img)
        manual_img = zero_pad_and_crop(manual_img)
        manual_img = Image.fromarray(manual_img)
        _ = np.random.randint(augment_policy.nsubpolicy)
        transformations = augment_policy.subpolicies[1]
        for t in transformations:
            if np.random.random() < t[1]:
                if len(t) == 3:
                    manual_img = t[0](manual_img, t[2])
                else:
                    manual_img = t[0](manual_img, t[2], t[3])
        manual_img = np.array(manual_img.getdata()).reshape(self.img_shape) / 255
        manual_img = cutout_numpy(manual_img)
        assert np.all(abs(aug_img - manual_img) < 1e-3)

    def test_svhn_applytransform(self):
        augment_policy = AugmentationPolicy(dataset="svhn")
        magnitude = 8
        autocontrast_img = augment_policy.apply_transform(self.pil_img, autocontrast, 1, magnitude)
        autocontrast_img = np.array(autocontrast_img.getdata()).reshape(self.img_shape)
        autocontrast_manual = np.array(autocontrast(self.pil_img, magnitude).getdata()).reshape(self.img_shape)
        assert np.all(autocontrast_img == autocontrast_manual)
        id_img = augment_policy.apply_transform(self.pil_img, autocontrast, 0, magnitude)
        id_img = np.array(id_img.getdata()).reshape(self.img_shape)
        assert np.all(id_img == self.np_img)
        np.random.seed(0)
        shear_img = augment_policy.apply_transform(self.pil_img, shear, 1, magnitude, "x")
        shear_img = np.array(shear_img.getdata()).reshape(self.img_shape)
        np.random.seed(0)
        shear_manual = np.array(shear(self.pil_img, magnitude, "x").getdata()).reshape(self.img_shape)
        assert np.all(abs(shear_img - shear_manual) < 1e-3)

    def test_svhn_call(self):
        augment_policy = AugmentationPolicy(dataset="svhn")
        np.random.seed(1)
        aug_img = augment_policy.call(self.tf_img, test=True)
        np.random.seed(1)
        manual_img = flip(self.np_img)
        manual_img = zero_pad_and_crop(manual_img)
        manual_img = Image.fromarray(manual_img)
        _ = np.random.randint(augment_policy.nsubpolicy)
        transformations = augment_policy.subpolicies[0]
        for t in transformations:
            if np.random.random() < t[1]:
                if len(t) == 3:
                    manual_img = t[0](manual_img, t[2])
                else:
                    manual_img = t[0](manual_img, t[2], t[3])
        manual_img = np.array(manual_img.getdata()).reshape(self.img_shape) / 255
        manual_img = cutout_numpy(manual_img)
        assert np.all(abs(aug_img - manual_img) < 1e-3)

    def test_imagenet_applytransform(self):
        augment_policy = AugmentationPolicy(dataset="imagenet")
        magnitude = 8
        autocontrast_img = augment_policy.apply_transform(self.pil_img, autocontrast, 1, magnitude)
        assert np.all(autocontrast_img == autocontrast(self.pil_img, magnitude))
        id_img = augment_policy.apply_transform(self.pil_img, autocontrast, 0, magnitude)
        id_img = np.array(id_img.getdata()).reshape(self.img_shape)
        assert np.all(id_img == self.np_img)
        np.random.seed(0)
        shear_img = augment_policy.apply_transform(self.pil_img, shear, 1, magnitude, "x")
        shear_img = np.array(shear_img.getdata()).reshape(self.img_shape)
        np.random.seed(0)
        assert np.all(abs(shear_img - shear(self.pil_img, magnitude, "x")) < 1e-3)

    def test_imagenet_call(self):
        augment_policy = AugmentationPolicy(dataset="imagenet")
        np.random.seed(0)
        aug_img = augment_policy.call(self.tf_img, test=True)
        np.random.seed(0)
        manual_img = flip(self.np_img)
        manual_img = zero_pad_and_crop(manual_img)
        manual_img = Image.fromarray(manual_img)
        _ = np.random.randint(augment_policy.nsubpolicy)
        transformations = augment_policy.subpolicies[18]
        for t in transformations:
            if np.random.random() < t[1]:
                manual_img = t[0](manual_img, t[2])
        manual_img = np.array(manual_img.getdata()).reshape(self.img_shape) / 255
        manual_img = cutout_numpy(manual_img)
        assert np.all(abs(aug_img - manual_img) < 1e-3)

    def test_nopolicy(self):
        augment_policy = AugmentationPolicy(dataset=None)
        np.random.seed(0)
        aug_img = augment_policy.call(self.tf_img)
        np.random.seed(0)
        manual_img = flip(self.np_img)
        manual_img = zero_pad_and_crop(manual_img)
        manual_img = Image.fromarray(manual_img)
        manual_img = np.array(manual_img.getdata()).reshape(self.img_shape) / 255
        manual_img = cutout_numpy(manual_img)
        assert np.all(abs(aug_img - manual_img) < 1e-3)
