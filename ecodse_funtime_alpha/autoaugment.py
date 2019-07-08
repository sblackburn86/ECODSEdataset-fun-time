import numpy as np
import PIL
import tensorflow as tf

from PIL import ImageOps


def pil_wrap(img):
    """Convert the img tensor to a PIL Image

    Parameters
    ----------
    img : tf.tensor
        image to wrap to PIL

    Returns
    -------
    PIL image
        image in PIL format
    """
    return PIL.Image.fromarray(np.uint8(img.numpy() * 255))


def pil_unwrap(pil_img):
    """Convert the 'pil_img' image to a tensorflow tensor

    Parameters
    ----------
    pil_img : PIL image
        image to convert

    Returns
    -------
    tf.tensor
        image as a tensorflow tensor
    """
    pic_array = (np.array(pil_img.getdata()).reshape((256, 256, 3)) / 255)
    return tf.convert_to_tensor(pic_array)


def autocontrast(img, magnitude=None):
    """apply autocontrast to image

    Parameters
    ----------
    img : tf.tensor
        input image
    magnitude : int, optional
        amplitude of the transformation, unused here, by default: None

    Returns
    -------
    tf.tensor
        image enhanced with autocontrast
    """
    return pil_unwrap(ImageOps.autocontrast(pil_wrap(img)))


def brightness(img, magnitude):
    """enhance brightness

    Parameters
    ----------
    img : tf.tensor
        input image as tensorflow tensor
    magnitude : int
        brightness magnitude, between 0 and 10

    Returns
    -------
    tf.tensor
        transformed image
    """
    magnitude = magnitude * 1.8 / 10 + 0.1
    return pil_unwrap(PIL.ImageEnhance.Brightness(pil_wrap(img)).enhance(magnitude))


def color(img, magnitude):
    """enhance colors

    Parameters
    ----------
    img : tf.tensor
        input image as tensorflow tensor
    magnitude : int
        color change magnitude, between 0 and 10

    Returns
    -------
    tf.tensor
        transformed image
    """
    magnitude = magnitude * 1.8 / 10 + 0.1
    return pil_unwrap(PIL.ImageEnhance.Color(pil_wrap(img)).enhance(magnitude))


def contrast(img, magnitude):
    """increase image contrast

    Parameters
    ----------
    img : tf.tensor
        input image
    magnitude : int
        constrast magnitude between 0 and 10

    Returns
    -------
    tf.tensor
        transformed image
    """
    # rescale magnitude
    magnitude = magnitude * 1.8 / 10 + 0.1
    return pil_unwrap(PIL.ImageEnhance.Contrast(pil_wrap(img)).enhance(magnitude))


def equalize(img, magnitude=None):
    """equalize image

    Parameters
    ----------
    img : tf.tensor
        input image
    magnitude : int, optional
        amplitude of the transformation, unused here, by default: None

    Returns
    -------
    tf.tensor
        equalized image
    """
    return pil_unwrap(ImageOps.equalize(pil_wrap(img)))


def invert(img, magnitude=None):
    """inverts colors

    Parameters
    ----------
    img : tf.tensor
        input image
    magnitude : int, optional
        amplitude of the transformation, unused here, by default: None

    Returns
    -------
    tf.tensor
        transformed image
    """
    return pil_unwrap(ImageOps.invert(pil_wrap(img)))


def posterize(img, magnitude):
    """posterize image

    Parameters
    ----------
    img : tf.tensor
        input image
    magnitude : int
        level of posterize between 0 and 10

    Returns
    -------
    tf.tensor
        posterized image as tensorflow tensor
    """
    magnitude = int(magnitude * 4 / 10)
    return pil_unwrap(ImageOps.posterize(pil_wrap(img), 4 - magnitude))


def rotate(img, magnitude):
    """rotate image by -30 to 30 based on magnitude

    Parameters
    ----------
    img : tf.tensor
        input image to rotate
    magnitude : int
        rotation angle by increments of 3 degrees

    Returns
    -------
    tf.tensor
        rotated image as a tensorflow tensor
    """
    magnitude *= 30 / 10
    if np.random.random() > 0.5:
        magnitude = -magnitude
    return pil_unwrap(pil_wrap(img).rotate(magnitude))


def sharpness(img, magnitude):
    """increase image contrast

    Parameters
    ----------
    img : tf.tensor
        input image
    magnitude : int
        sharpness magnitude between 0 and 10

    Returns
    -------
    tf.tensor
        transformed image
    """
    # rescale sharpness
    magnitude = magnitude * 1.8 / 10 + 0.1
    return pil_unwrap(PIL.ImageEnhance.Sharpness(pil_wrap(img)).enhance(magnitude))


def shear(img, magnitude, axis="x"):
    """
    shear an image along an axis based on magnitude

    Parameters
    ----------
    img : tf.tensor
        input image to shear
    magnitude : int
        magnitude of the shear
    axis: string, optional
        along x or y axis, by default x

    Returns
    -------
    tf.tensor
        transformed image
    """

    assert axis in ["x", "y"]
    magnitude *= 0.3 / 10
    if np.random.random() > 0.5:
        magnitude = -magnitude
    img = pil_wrap(img)
    if axis == "x":
        # do shearX
        return pil_unwrap(img.transform((256, 256), PIL.Image.AFFINE, (1, magnitude, 0, 0, 1, 0)))
    else:
        # do shearY
        return pil_unwrap(img.transform((256, 256), PIL.Image.AFFINE, (1, 0, 0, magnitude, 1, 0)))


def solarize(img, magnitude):
    """applies PIL solarize to image

    Parameters
    ----------
    img : tf.tensor
        input image
    magnitude : int
        magnitude of the solarize operation as an int between 0 and 10

    Returns
    -------
    tf.tensor
        transformed image as a tensorflow tensor
    """
    magnitude *= 256 / 10
    return pil_unwrap(ImageOps.solarize(pil_wrap(img), 256 - magnitude))


def translate(img, magnitude, axis="x"):
    """translate image by number of pixels equal to magnitude along x or y axis

    Parameters
    ----------
    img : tf.tensor
        input image
    magnitude : int
        number of pixels to translate
    axis : str, optional
        axis to translate, x or y, by default "x"

    Returns
    -------
    tf.tensor
        translated image
    """
    assert isinstance(magnitude, int)
    assert axis in ["x", "y"]
    if np.random.random() > 0.5:
        magnitude = -magnitude
    img = pil_wrap(img)
    if axis == "x":
        # do translateX
        return pil_unwrap(img.transform((256, 256), PIL.Image.AFFINE, (1, 0, magnitude, 0, 1, 0)))
    else:
        # do translateY
        return pil_unwrap(img.transform((256, 256), PIL.Image.AFFINE, (1, 0, 0, 0, 1, magnitude)))


class CIFAR10_policy(object):
    """
    Auto-augment policy for CIFAR10

    Attributes
    ----------
    subpolicies : list of tuples
        each element if a set of 2 operations for image augmentation representing the 25 optimal policies for CIFAR10

    Methods
    -------
    apply_transformation:
        apply a given transformation to an image
    call:
        transform an image according to a random sub-policy
        Usage example: augmentation_policy.call(image)
    """

    def __init__(self):
        """
        Class Constructor
        """
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

    def apply_transform(self, img, transformation, prob, magnitude=None, axis=None):
        """apply a transformation to an image with probability prob

        Parameters
        ----------
        img : tf.tensor
            image to transform as a tensorflow tensor
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
        tf tensor
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

    def call(self, image):
        """
        return an augmented image according to the policy

        Parameters
        ----------
        image : tf.tensor
            image stored as a tensor

        Returns
        -------
        tf.tensor
            transformed image
        """
        # first, choose a random sub-policy to apply
        subpol = self.subpolicies[np.random.randint(len(self.subpolicies))]
        for operation in subpol:
            image = self.apply_transform(image, *operation)
        return image


class SVHN_policy(object):
    """
    Auto-augment policy for SVHN

    Attributes
    ----------
    subpolicies : list of tuples
        each element if a set of 2 operations for image augmentation representing the 25 optimal policies for SVHN

    Methods
    -------
    apply_transformation:
        apply a given transformation to an image
    call:
        transform an image according to a random sub-policy
        Usage example: augmentation_policy.call(image)
    """

    def __init__(self):
        """
        Class Constructor
        """
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

    def apply_transform(self, img, transformation, prob, magnitude=None, axis=None):
        """apply a transformation to an image with probability prob

        Parameters
        ----------
        img : tf.tensor
            image to transform as a tensorflow tensor
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
        tf tensor
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

    def call(self, image):
        """
        return an augmented image according to the policy

        Parameters
        ----------
        image : tf.tensor
            image stored as a tensor

        Returns
        -------
        tf.tensor
            transformed image
        """
        # first, choose a random sub-policy to apply
        subpol = self.subpolicies[np.random.randint(len(self.subpolicies))]
        for operation in subpol:
            image = self.apply_transform(image, *operation)
        return image
