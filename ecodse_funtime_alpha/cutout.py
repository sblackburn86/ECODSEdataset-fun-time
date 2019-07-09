import numpy as np
import tensorflow as tf


def create_cutout_mask(img_height, img_width, num_channels, size):
    """creates a zero mask used for cutout of shape img_height x img_width

    Parameters
    ----------
    img_height : int
        height of image
    img_width : int
        width of image
    num_channels : int
        number of channels of the image
    size : int
        size of the zero mask

    Returns
    -------
    mask: np.array
        mask of shape img_height x img_width with all 1s, except for a square of zeros of shape size x size
    upper_coord: int
        upper coordinates of the patch of zeros
    lower_coord: int
        lower coordinates of the patch of zeros

    """
    assert img_height == img_width

    # sample center where cutout mask will be applied
    height_loc = np.random.randint(low=0, high=img_height)
    width_loc = np.random.randint(low=0, high=img_width)

    # determine upper right and lower left corners of patch
    upper_coord = (max(0, height_loc - size // 2), max(0, width_loc - size // 2))
    lower_coord = (min(img_height, height_loc + size // 2), min(img_width, width_loc + size // 2))
    mask_height = lower_coord[0] - upper_coord[0]
    mask_width = lower_coord[1] - upper_coord[1]

    assert mask_height > 0
    assert mask_width > 0

    mask = np.ones((img_height, img_width, num_channels))
    zeros = np.zeros((mask_height, mask_width, num_channels))
    mask[upper_coord[0]:lower_coord[0], upper_coord[1]:lower_coord[1], :] = (zeros)
    return mask, upper_coord, lower_coord


def cutout_numpy(img, size=16):
    """apply cutout with mask of shape size x size

    Parameters
    ----------
    img : np.array
        image that cutout will be applied to
    size : int, optional
        size of the 0 cutout patch, by default 16

    Returns
    -------
    np.array
        image with a cutout section
    """
    mask, _, _ = create_cutout_mask(*img.shape, size)
    return img * mask


def cutout_tf(img, size=16):
    """apply cutout to a tensorflow image

    Parameters
    ----------
    img : tf.tensor
        input image
    size : int
        size of the cutout patch, by default 16

    Returns
    -------
    tf.tensor
        tensorflow tensor with a cutout section
    """
    return tf.convert_to_tensor(cutout_numpy(img.numpy(), size))
