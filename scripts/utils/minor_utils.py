"""Helper functions for Pillow visualization"""

from PIL import Image
import numpy as np


def make_mask_visible(mask):
    """Converts Pillow Image representing binary mask to helpful Pillow Image.

    Args:
        mask (PIL.Image): Pillow image of binary mask

    Returns:
        PIL.Image: Pillow image
    """
    return Image.fromarray((np.asarray(mask)*255).astype(np.uint8))


def make_mask_arr_visible(mask_arr):
    """Converts numpy array representing binary mask to helpful Pillow Image.

    Args:
        mask_arr (ndarray): numpy array of binary mask

    Returns:
        PIL.Image: Pillow image
    """
    return Image.fromarray((mask_arr*255).astype(np.uint8))


def min_max_norm(arr):
    """Min-max normalization

    Args:
        arr (ndarray): numpy array

    Returns:
        ndarray: min-max normalized numpy array
    """
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr
