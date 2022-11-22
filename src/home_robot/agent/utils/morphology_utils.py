import torch


def binary_dilation(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
    """
    return torch.clamp(
        torch.nn.functional.conv2d(binary_image, kernel, padding=kernel.shape[-1] // 2),
        0,
        1,
    )


def binary_erosion(binary_image, kernel):
    """
    Arguments:
        binary_image: binary image tensor of shape (bs, 1, H1, W1)
        kernel: binary structuring element tensor of shape (1, 1, H2, W2)

    Returns:
        binary image tensor of the same shape as input
    """
    return 1 - torch.clamp(
        torch.nn.functional.conv2d(
            1 - binary_image, kernel, padding=kernel.shape[-1] // 2
        ),
        0,
        1,
    )


def binary_opening(binary_image, kernel):
    return binary_dilation(binary_erosion(binary_image, kernel), kernel)


def binary_closing(binary_image, kernel):
    return binary_erosion(binary_dilation(binary_image, kernel), kernel)


def binary_denoising(binary_image, kernel):
    return binary_opening(binary_closing(binary_image, kernel), kernel)
