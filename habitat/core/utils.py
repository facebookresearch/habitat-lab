from typing import List

import numpy as np


# TODO(akadian): numpy does not support type hinting organically:
# issue: https://github.com/numpy/numpy/issues/7370


def tile_images(images: List[np.ndarray]) -> np.ndarray:
    r"""
    :param images: list of images where each image has dimension
                   (height x width x channels)
    :return: tiled image (new_height x width x channels)
    """
    assert len(images) > 0, "empty list of images"
    images = np.asarray(images)
    n_images, height, width, n_channels = images.shape  # type: ignore
    new_height = int(np.ceil(np.sqrt(n_images)))
    new_width = int(np.ceil(float(n_images) / new_height))
    # pad with empty images to complete the rectangle
    images = np.array(
        list(images)
        + [images[0] * 0 for _ in range(n_images, new_height * new_width)]
    )
    # img_HWhwc
    out_image = images.reshape(
        new_height, new_width, height, width, n_channels  # type: ignore
    )
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(
        new_height * height, new_width * width, n_channels
    )
    return out_image
