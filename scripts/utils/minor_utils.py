

from PIL import Image
import numpy as np


def make_mask_visible(mask):
    return Image.fromarray((np.asarray(mask)*255).astype(np.uint8))


def make_mask_arr_visible(mask_arr):
    return Image.fromarray((mask_arr*255).astype(np.uint8))


def min_max_norm(arr):
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr
