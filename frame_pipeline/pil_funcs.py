"""
Functions for working with PIL.Image.Image objects.
"""

import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..utility.builtin_classes.iterables import split_chunks
from ..utility.general import standardize_padding


def ensure_pil_image(image_or_path, array_is_bgr=False):
    """
    Ensures that the specified object is a PIL.Image.Image object. If the object is a path to an image file, the image
    will be loaded from the file. If the object is already a PIL.Image.Image object, it will be returned as is. If the
    object is a numpy array, it will be converted to a PIL.Image.Image object.
    This function is useful as an adapter for functions that accept either a PIL.Image.Image object or a path to an
    image file.

    :param image_or_path: A path to an image file or a PIL.Image.Image object or a numpy array
    :type image_or_path: str or PIL.Image.Image or numpy.ndarray
    :param array_is_bgr: If the input is an array, whether to assume that the colors in the channels are inversed
    (relative to Pillows RGB standard). Inverts the channels before converting to PIL.Image.Image object. Default is
    False.
    :type array_is_bgr: bool
    :return: A PIL.Image.Image object
    :rtype: PIL.Image.Image
    """
    if isinstance(image_or_path, (str, pathlib.Path)):
        return Image.open(image_or_path)
    elif isinstance(image_or_path, Image.Image):
        return image_or_path
    elif isinstance(image_or_path, np.ndarray):
        if (image_or_path.ndim==3) and array_is_bgr:
            image_or_path = image_or_path.copy()
            image_or_path[:, :, [0, 1, 2]] = image_or_path[:, :, [2, 1, 0]]

        return Image.fromarray(image_or_path)
    else:
        raise TypeError("image_or_path must be a path to an image file, a PIL.Image.Image object, or a numpy array!")


def create_blank_frame_like(pil_image, color=(255, 255, 255)):
    """
    Creates a blank frame with the same size as the specified image, filled with the specified color.

    :param pil_image: A PIL.Image.Image object
    :type pil_image: PIL.Image.Image
    :param color: The color to fill the frame with
    :type color: tuple
    :return: A blank frame with the same size as the specified image, filled with the specified color
    :rtype: PIL.Image.Image
    """
    return Image.new('RGB', pil_image.size, color)


def pad_image(pil_image, padding, pad_color=(255, 255, 255)):
    """
    Pads an image with the specified padding and pad_color.

    :param pil_image: A PIL.Image.Image object
    :type pil_image: PIL.Image.Image
    :param padding: A tuple or list of length 1, 2, or 4, specifying the padding to apply. If the length is 1, the same
                    padding is applied to all sides. If the length is 2, the first value is applied to the top and
                    bottom, and the second value is applied to the left and right sides. If the length is 4, the
                    values are applied to the left, top, right, and bottom sides, respectively.
    :type padding: tuple or list
    :param pad_color: The color to pad the frame with
    :type pad_color: tuple or list
    :return: The padded image
    :rtype: PIL.Image.Image
    """
    padding = standardize_padding(padding)

    padded_image = Image.new('RGB', (pil_image.size[0] + padding[0] + padding[2],
                                     pil_image.size[1] + padding[1] + padding[3]), color=pad_color)
    padded_image.paste(im=pil_image, box=(padding[0], padding[1]))
    return padded_image


def resize_with_preserved_aspect_ratio(pil_image, new_size, pad_color=(255, 255, 255), **resize_kwargs):
    """
    Resizes an image to the specified size while preserving the aspect ratio. The image is resized so that it fits
    within the specified size, and the remaining space is filled with the specified pad_color.

    :param pil_image: A PIL.Image.Image object
    :type pil_image: PIL.Image.Image
    :param new_size: The new size of the image
    :type new_size: tuple of int
    :param pad_color: The color to pad the frame with
    :type pad_color: tuple
    :param resize_kwargs: Additional keyword arguments to pass to the resize function
    :type resize_kwargs: dict
    :return: The resized image
    :rtype: PIL.Image.Image
    """
    image_size = pil_image.size
    resize_factor = min([new_size[0] / image_size[0], new_size[1] / image_size[1]])
    resized_image = pil_image.resize((int(image_size[0] * resize_factor), int(image_size[1] * resize_factor)),
                                     **resize_kwargs)
    width_pad = (new_size[0] - resized_image.size[0]) // 2
    height_pad = (new_size[1] - resized_image.size[1]) // 2
    padded_image = pad_image(resized_image, (width_pad, height_pad, width_pad, height_pad),
                             pad_color=pad_color)
    return padded_image


def add_text(pil_image, text_string, relative_xy=(0.05, 0.9), color=(255, 255, 255), anchor="ma"):
    text_image = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_image)

    width, height = pil_image.size
    font = ImageFont.truetype("arial.ttf", 40)
    text_x, text_y = width * relative_xy[0], height * relative_xy[1]
    draw.text((text_x, text_y), text_string, fill=color, font=font, anchor=anchor)

    output_image = pil_image.copy()
    output_image.paste(text_image, (0, 0), mask=text_image)
    return output_image


def merge_images(*images, positions=None, pad_color=(255, 255, 255)):
    """
    A function that takes an arbitrary number of PIL.Image.Image objects and merges them into a single image. The
    images are inserted into the resulting image at the given coordinates, with the top left corner of the image at the
    given coordinates. The resulting image is returned. If there is empty space due to the size of the images, then the
    empty space is filled with the specified pad_color. If all input images have the same mode, the resulting image will
    have the same mode. Otherwise, the resulting image will have RGBA mode.

    :param images: A number of PIL.Image.Image objects
    :type images: PIL.Image.Image
    :param positions: A list of (x, y) tuples, specifying the coordinates of the top left corner of each image. If None,
                      the images are inserted one after the other, from left to right.
    :type positions: list of tuple
    :param pad_color: The color to pad the frame with, in case the images have different sizes
    :type pad_color: int or float or tuple
    :return: The merged image
    :rtype: PIL.Image.Image
    """
    if len(images) == 0:
        raise ValueError("At least one image must be specified!")
    if positions is not None and len(images) != len(positions):
        raise ValueError("The number of images and the number of positions must be the same!")
    if positions is None:
        _image_widths = [image.size[0] for image in images]
        positions = [(sum(_image_widths[:i]), 0) for i in range(len(_image_widths))]

    _unique_input_modes = set([image.mode for image in images])
    if len(_unique_input_modes) == 1:
        return_mode = _unique_input_modes.pop()
    else:
        return_mode = 'RGBA'

    return_image = Image.new(return_mode,
                             (max([position[0] + image.size[0] for image, position in zip(images, positions)]),
                              max([position[1] + image.size[1] for image, position in zip(images, positions)])),
                             pad_color)
    for image, position in zip(images, positions):
        return_image.paste(image, box=position, mask=image if image.mode == 'RGBA' else None)
    return return_image


def concat_images(*images, append_bottom=False, pad_color=(255, 255, 255)):
    """
    A convenience wrapper for merge_pil_images that takes an arbitrary number of PIL.Image.Image objects and merges
    them into a single image. The images are merged side by side, unless append_bottom is True, in which case the
    images are merged one on top of the other. The resulting image is returned. If the images have different sizes, the
    smaller image is padded with the specified pad_color.

    :param images: A number of PIL.Image.Image objects
    :type images: PIL.Image.Image
    :param append_bottom: Whether to append the images to the bottom of each other. If False, the images will be
    appended to the right of each other.
    :type append_bottom: bool
    :param pad_color: The color to pad the frame with, in case the images have different sizes
    :type pad_color: tuple
    :return: The merged image
    :rtype: PIL.Image.Image
    """
    if len(images) == 0:
        raise ValueError("At least one image must be specified!")
    else:
        if append_bottom:
            positions = [(0, sum([image.size[1] for image in images[:i]])) for i in range(len(images))]
        else:
            positions = None
        return merge_images(*images, positions=positions, pad_color=pad_color)


def stitch_image_list(*images, rows=None, columns=None, pad_color=(255, 255, 255)):
    """
    A function that takes a list or tuple of PIL.Image.Image objects and stitches them together into a single image. The
    images are stitched together in a grid with the specified shape. The images are stitched together row by row, and
    then the rows are stitched together. The resulting image is returned.
    If the product of rows and columns is smaller than the number of images, the remaining images are ignored.
    If the product of rows and columns is larger than the number of images, the resulting image will only have as many
    rows and columns as there are images. Completely empty rows and columns are not added.

    :param images: A list or tuple of PIL.Image.Image objects
    :type images: list of PIL.Image.Image or tuple of PIL.Image.Image
    :param rows: The number of rows in the grid. If None, the number of rows is calculated automatically.
    :type rows: int or None
    :param columns: The number of columns in the grid. If None, the number of columns is calculated automatically.
    :type columns: int or None
    :param pad_color: The color to pad the frame with, in case the images have different sizes
    :type pad_color: tuple
    :return: The stitched image
    :rtype: PIL.Image.Image
    """
    if rows is None and columns is None:
        raise ValueError("At least one of rows and columns must be specified!")
    if rows is None:
        rows = len(images) // columns + (len(images) % columns > 0)
    if columns is None:
        columns = len(images) // rows + (len(images) % rows > 0)

    _row_images = []

    for _, image_row in zip(range(rows), split_chunks(images, chunk_size=columns)):
        _row_images.append(concat_images(*image_row, append_bottom=False, pad_color=pad_color))

    merged_image = concat_images(*_row_images, append_bottom=True, pad_color=pad_color)
    return merged_image


def read_gif_frames(gif_image):
    """
    A function that takes an animated GIF and returns a list of frames. It also resets the GIF to the frame it was at
    before the function was called.

    :param gif_image: An animated GIF
    :type gif_image: PIL.Image.Image
    :return: A list of frames
    :rtype: list[PIL.Image.Image]
    """
    frame_list = []
    current_index = gif_image.tell()
    gif_image.seek(0)
    try:
        while 1:
            gif_image.seek(gif_image.tell() + 1)
            frame_list.append(gif_image.copy())
    except EOFError:
        pass  # end of sequence
    gif_image.seek(current_index)
    return frame_list


def stitch_animated_gif_list(*gif_images, rows=None, columns=None, pad_color=(255, 255, 255)):
    """
    A function that takes an arbitrary number of animated GIFs and stitches them together into a stack of frames.
    This first extracts lists of frames from each GIF, then stitches the frames together using stitch_image_list, and
    returns the resulting stitched GIF frames as a list of PIL.Image.Image objects.
    For further documentation, see stitch_image_list.

    :param gif_images: An arbitrary number of animated GIFs
    :type gif_images: PIL.Image.Image
    :param rows: The number of rows in the grid. If None, the number of rows is calculated automatically.
    :type rows: int or None
    :param columns: The number of columns in the grid. If None, the number of columns is calculated automatically.
    :type columns: int or None
    :param pad_color: The color to pad the frame with, in case the images have different sizes
    :type pad_color: tuple
    :return: A list of stitched GIF frames
    :rtype: list[PIL.Image.Image]
    """
    gif_frames = [read_gif_frames(gif_image) for gif_image in gif_images]
    concurrent_frames = list(zip(*gif_frames))
    merged_frames = [stitch_image_list(*frames, rows=rows, columns=columns, pad_color=pad_color) for frames in
                     concurrent_frames]

    return merged_frames