import numpy as np
import PIL
import tensorflow as tf

from PIL import Image
from PIL import ImageOps

from ecodse_funtime_alpha.cutout import cutout_numpy


def autocontrast(img, magnitude=None):
    """apply autocontrast to image

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int, optional
        amplitude of the transformation, unused here, by default: None

    Returns
    -------
    PIL.Image
        image enhanced with autocontrast
    """
    return ImageOps.autocontrast(img)


def brightness(img, magnitude):
    """enhance brightness

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int
        brightness magnitude, between 0 and 10

    Returns
    -------
    PIL.Image
        transformed image
    """
    magnitude = magnitude * 1.8 / 10 + 0.1
    return PIL.ImageEnhance.Brightness(img).enhance(magnitude)


def color(img, magnitude):
    """enhance colors

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int
        color change magnitude, between 0 and 10

    Returns
    -------
    PIL.Image
        transformed image
    """
    magnitude = magnitude * 1.8 / 10 + 0.1
    return PIL.ImageEnhance.Color(img).enhance(magnitude)


def contrast(img, magnitude):
    """increase image contrast

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int
        constrast magnitude between 0 and 10

    Returns
    -------
    PIL.Image
        transformed image
    """
    # rescale magnitude
    magnitude = magnitude * 1.8 / 10 + 0.1
    return PIL.ImageEnhance.Contrast(img).enhance(magnitude)


def equalize(img, magnitude=None):
    """equalize image

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int, optional
        amplitude of the transformation, unused here, by default: None

    Returns
    -------
    PIL.Image
        equalized image
    """
    return ImageOps.equalize(img)


def flip(img):
    """flip the input image horizontally with 50% probability

    Parameters
    ----------
    img : np.array
        input image

    Returns
    -------
    np.array
        transformed image
    """
    if np.random.random() > 0.5:
        img = np.fliplr(img)
    return img


def invert(img, magnitude=None):
    """inverts colors

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int, optional
        amplitude of the transformation, unused here, by default: None

    Returns
    -------
    PIL.Image
        transformed image
    """
    return ImageOps.invert(img)


def posterize(img, magnitude):
    """posterize image

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int
        level of posterize between 0 and 10

    Returns
    -------
    PIL.Image
        posterized image
    """
    magnitude = 4 - int(magnitude * 4 / 10)
    return ImageOps.posterize(img, magnitude)


def rotate(img, magnitude):
    """rotate image by -30 to 30 based on magnitude

    Parameters
    ----------
    img : PIL.Image
        input image to rotate
    magnitude : int
        rotation angle by increments of 3 degrees

    Returns
    -------
    PIL.Image
        rotated image
    """
    magnitude *= 30 / 10
    if np.random.random() > 0.5:
        magnitude = -magnitude
    return img.rotate(magnitude)


def sharpness(img, magnitude):
    """increase image contrast

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int
        sharpness magnitude between 0 and 10

    Returns
    -------
    PIL.Image
        transformed image
    """
    # rescale sharpness
    magnitude = magnitude * 1.8 / 10 + 0.1
    return PIL.ImageEnhance.Sharpness(img).enhance(magnitude)


def shear(img, magnitude, axis="x"):
    """
    shear an image along an axis based on magnitude

    Parameters
    ----------
    img : PIL.Image
        input image to shear
    magnitude : int
        magnitude of the shear
    axis: string, optional
        along x or y axis, by default x

    Returns
    -------
    PIL.Image
        transformed image
    """

    assert axis in ["x", "y"]
    magnitude *= 0.3 / 10
    if np.random.random() > 0.5:
        magnitude = -magnitude
    if axis == "x":
        # do shearX
        return img.transform((256, 256), PIL.Image.AFFINE, (1, magnitude, 0, 0, 1, 0))
    else:
        # do shearY
        return img.transform((256, 256), PIL.Image.AFFINE, (1, 0, 0, magnitude, 1, 0))


def solarize(img, magnitude):
    """applies PIL solarize to image

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int
        magnitude of the solarize operation as an int between 0 and 10

    Returns
    -------
    PIL.Image
        transformed image
    """
    magnitude = 256 - magnitude * 256 / 10
    return ImageOps.solarize(img, magnitude)


def translate(img, magnitude, axis="x"):
    """translate image by number of pixels equal to magnitude along x or y axis

    Parameters
    ----------
    img : PIL.Image
        input image
    magnitude : int
        number of pixels to translate
    axis : str, optional
        axis to translate, x or y, by default "x"

    Returns
    -------
    PIL.Image
        translated image
    """
    assert isinstance(magnitude, int)
    assert axis in ["x", "y"]
    if np.random.random() > 0.5:
        magnitude = -magnitude
    if axis == "x":
        # do translateX
        return img.transform((256, 256), PIL.Image.AFFINE, (1, 0, magnitude, 0, 1, 0))
    else:
        # do translateY
        return img.transform((256, 256), PIL.Image.AFFINE, (1, 0, 0, 0, 1, magnitude))


def zero_pad_and_crop(img, amount=4):
    """zero padding by a number of pixels equal to amount on each side, then take a random crop

    Parameters
    ----------
    img : np.array
        input image
    amount : int, optional
        amount of zeros to pad by default 4

    Returns
    -------
    np.array
        transformed image
    """
    padded_img = np.zeros((img.shape[0] + amount * 2, img.shape[1] + amount * 2, img.shape[2]))
    padded_img[amount:img.shape[0] + amount, amount:img.shape[1] + amount, :] = img
    top = np.random.randint(low=0, high=2 * amount)
    left = np.random.randint(low=0, high=2 * amount)
    return np.uint8(padded_img[top:top + img.shape[0], left:left + img.shape[1], :])


class AugmentationPolicy(object):
    """
    Auto-augment policy

    Attributes
    ----------
    subpolicies : list of tuples
        each element if a set of 2 operations for image augmentation representing the 25 optimal policies for CIFAR10
    dataset : str
        name of the dataset the chosen policy was trained on

    Methods
    -------
    apply_transformation:
        apply a given transformation to an image
    call:
        transform an image according to a random sub-policy
        Usage example: augmentation_policy.call(image)
    """

    def __init__(self, dataset="cifar10"):
        """
        Class Constructor

        Parameters
        ----------
        dataset : str, optional
            which policy to use, by default cifar10
            options are cifar10, imagenet and svhn

        """

        self.dataset = dataset

        if dataset == "cifar10":
            self.subpolicies = [
                [(invert, 0.1, 7), (contrast, 0.2, 6)],
                [(rotate, 0.7, 2), (translate, 0.3, 9, "x")],
                [(sharpness, 0.8, 1), (sharpness, 0.9, 3)],
                [(shear, 0.5, 8, "y"), (translate, 0.7, 9, "y")],
                [(autocontrast, 0.5, 8), (equalize, 0.9, 2)],
                [(shear, 0.2, 7, "y"), (posterize, 0.3, 7)],
                [(color, 0.4, 3), (brightness, 0.6, 7)],
                [(sharpness, 0.3, 9), (brightness, 0.7, 9)],
                [(equalize, 0.6, 5), (equalize, 0.5, 1)],
                [(contrast, 0.6, 7), (sharpness, 0.6, 5)],
                [(color, 0.7, 7), (translate, 0.5, 8, "x")],
                [(equalize, 0.3, 7), (autocontrast, 0.4, 8)],
                [(translate, 0.4, 3, "y"), (sharpness, 0.2, 6)],
                [(brightness, 0.9, 6), (color, 0.2, 8)],
                [(solarize, 0.5, 2), (invert, 0, 3)],
                [(equalize, 0.2, 0), (autocontrast, 0.6, 0)],
                [(equalize, 0.2, 8), (equalize, 0.6, 4)],
                [(color, 0.9, 9), (equalize, 0.6, 6)],
                [(autocontrast, 0.8, 4), (solarize, 0.2, 8)],
                [(brightness, 0.1, 3), (color, 0.7, 0)],
                [(solarize, 0.1, 3), (autocontrast, 0.9, 3)],
                [(translate, 0.9, 9, "y"), (translate, 0.7, 9, "y")],
                [(autocontrast, 0.9, 2), (solarize, 0.8, 3)],
                [(equalize, 0.8, 8), (invert, 0.1, 3)],
                [(translate, 0.7, 9, "y"), (autocontrast, 0.9, 1)]
            ]

        elif dataset == "imagenet":
            self.subpolicies = [
                [(posterize, 0.4, 8), (rotate, 0.6, 9)],
                [(solarize, 0.6, 5), (autocontrast, 0.6, 5)],
                [(equalize, 0.8, 8), (equalize, 0.6, 3)],
                [(posterize, 0.6, 7), (posterize, 0.6, 6)],
                [(equalize, 0.4, 7), (solarize, 0.2, 4)],
                [(equalize, 0.4, 4, (rotate, 0.8, 8))],
                [(solarize, 0.6, 3), (equalize, 0.6, 7)],
                [(posterize, 0.8, 5), (equalize, 1, 2)],
                [(rotate, 0.2, 3), (solarize, 0.6, 8)],
                [(equalize, 0.6, 8), (posterize, 0.4, 6)],
                [(rotate, 0.8, 8), (color, 0.4, 0)],
                [(rotate, 0.4, 9), (equalize, 0.6, 2)],
                [(equalize, 0, 7), (equalize, 0.8, 8)],
                [(invert, 0.6, 4), (equalize, 1, 8)],
                [(color, 0.6, 4), (contrast, 1, 8)],
                [(rotate, 0.8, 8), (color, 1, 2)],
                [(color, 0.8, 8), (solarize, 0.8, 7)],
                [(sharpness, 0.4, 7), (invert, 0., 8)],
                [(shear, 0.6, 5, "x"), (equalize, 1, 9)],
                [(color, 0.4, 0), (equalize, 0.6, 3)],
                [(equalize, 0.4, 7), (solarize, 0.2, 4)],
                [(solarize, 0.6, 5), (autocontrast, 0.6, 5)],
                [(invert, 0.6, 4), (equalize, 1, 8)],
                [(color, 0.6, 4), (contrast, 1, 8)],
                [(equalize, 0.8, 8), (equalize, 0.6, 3)]
            ]

        elif dataset == "svhn":
            self.subpolicies = [
                [(shear, 0.9, 4, "x"), (invert, 0.2, 3)],
                [(shear, 0.9, 8, "y"), (invert, 0.7, 5)],
                [(equalize, 0.6, 5), (solarize, 0.6, 6)],
                [(invert, 0.9, 3), (equalize, 0.6, 3)],
                [(equalize, 0.6, 1), (rotate, 0.9, 3)],
                [(shear, 0.9, 4, "x"), (autocontrast, 0.8, 3)],
                [(shear, 0.9, 8, "y"), (invert, 0.4, 5)],
                [(shear, 0.9, 5, "y"), (solarize, 0.2, 6)],
                [(invert, 0.9, 6), (autocontrast, 0.8, 1)],
                [(equalize, 0.6, 3), (rotate, 0.9, 3)],
                [(shear, 0.9, 4, "x"), (solarize, 0.3, 3)],
                [(shear, 0.8, 8, "y"), (invert, 0.7, 4)],
                [(equalize, 0.9, 5), (translate, 0.6, 6, "y")],
                [(invert, 0.9, 4), (equalize, 0.6, 7)],
                [(contrast, 0.3, 3), (rotate, 0.8, 4)],
                [(invert, 0.8, 5), (translate, 0., 2, "y")],
                [(shear, 0.7, 6, "y"), (solarize, 0.4, 8)],
                [(invert, 0.6, 4), (rotate, 0.8, 4)],
                [(shear, 0.3, 7, "y"), (translate, 0.9, 3, "x")],
                [(shear, 0.1, 6, "x"), (invert, 0.6, 5)],
                [(solarize, 0.7, 2), (invert, 0.6, 5)],
                [(shear, 0.8, 4, "y"), (invert, 0.8, 8)],
                [(shear, 0.7, 9, "x"), (translate, 0.8, 3, "y")],
                [(shear, 0.8, 5, "y"), (autocontrast, 0.7, 3)],
                [(shear, 0.7, 2, "x"), (invert, 0.1, 5)]
            ]

        else:
            self.subpolicies = []

        self.nsubpolicy = len(self.subpolicies)

    def apply_transform(self, img, transformation, prob, magnitude=None, axis=None):
        """apply a transformation to an image with probability prob

        Parameters
        ----------
        img : np.array
            image to transform as a numpy array
        tranformation : function
            transformation to apply
        prob : float
            probability to apply the transformation
        magnitude : int, optional
            magnitude of the transformation, by default None
        axis : str, optional
            axis on which to apply transformation (translate or shear only), by default None

        Returns
        -------
        np array
            transformed image
        """
        assert prob <= 1 and prob >= 0
        if np.random.random() < prob:
            if axis:
                return transformation(img, magnitude, axis)
            else:
                return transformation(img, magnitude)
        else:
            return img

    def call(self, image, test=False):
        """
        return an augmented image according to the policy

        Parameters
        ----------
        image : tf.tensor
            image stored as a tensor

        test : bool
            if true, will use a fixed subpolicy, and not a random one

        Returns
        -------
        tf.tensor
            transformed image
        """
        # convert image to numpy
        image = np.uint8(image.numpy() * 255)

        # apply a standard preprocess to image
        image = flip(image)
        image = zero_pad_and_crop(image)

        # choose a random sub-policy
        if self.nsubpolicy > 0:
            subpol = self.subpolicies[np.random.randint(self.nsubpolicy)]
        else:
            subpol = []

        if test:
            subpol_idx = 1 if self.dataset == "cifar10" else 18 if self.dataset == "imagenet" else 0
            subpol = self.subpolicies[subpol_idx]

        # convert image to PIL format and apply sub-policy
        image = Image.fromarray(image)
        for operation in subpol:
            image = self.apply_transform(image, *operation)

        # convert to numpy array
        image = np.array(image.getdata()).reshape((256, 256, 3)) / 255

        # apply cutout to the image
        image = cutout_numpy(image)

        return tf.convert_to_tensor(image)
