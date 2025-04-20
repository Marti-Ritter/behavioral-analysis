import os

import numpy as np
import pandas as pd
from PIL import Image
from PIL.TiffImagePlugin import TiffImageFile
from PIL.TiffTags import TAGS
from tifffile import imwrite
from tqdm.auto import tqdm
from ..utility.builtin_classes.dicts import invert_flat_dict

from .pil_funcs import merge_images


def stitch_columbus_segments_to_tiffstack(index_csv_path, image_dir, correct_basic=False,
                                          out_dir=None, out_format="Well{well}Stack.tif", **basic_kwargs):
    """
    Stitches Columbus segments to a TIFF stack. The TIFF stack is saved in the specified output directory with the
    specified output format. The TIFF stack is saved in ImageJ format.
    Optionally, the images can be corrected using the BaSiC method. The documentation for basicpy can be found at
    https://basicpy.readthedocs.io/en/latest/index.html. The basicpy package must be installed to use this option.
    The package is only imported if this option is set to True. The BaSiC correction is then applied to each channel
    separately.

    :param index_csv_path: The path to the CSV file containing the Columbus index
    :type index_csv_path: str
    :param image_dir: The directory containing the Columbus images
    :type image_dir: str
    :param correct_basic: Whether to apply BaSiC correction to the images.
    :type correct_basic: bool
    :param out_dir: The directory to save the TIFF stack to. If None, the TIFF stack will be saved in the image
    directory
    :type out_dir: str
    :param out_format: The format of the output TIFF stack. The format should contain a single placeholder for the well
    name.
    :type out_format: str
    :param basic_kwargs: Keyword arguments to pass to the BaSiC constructor
    :type basic_kwargs: Any
    :return: None
    """

    if out_dir is None:
        out_dir = image_dir

    default_basic_kwargs = {"get_darkfield": True, "smoothness_flatfield": 1, "max_reweight_iterations": 20}
    for key, value in basic_kwargs.items():
        default_basic_kwargs[key] = value

    image_index_df = pd.read_csv(index_csv_path, sep=";")
    image_index_df["file_name"] = image_index_df["sourcefilename"].str.split("/").apply(pd.Series)[1]
    image_index_df["file_path"] = image_index_df["file_name"].apply(lambda x: os.path.join(image_dir, x))

    required_cols = ["WellName", "PositionX@um", "PositionY@um", "Plane", "ImageSizeX", "ImageSizeY", "Channel",
                     "file_path"]

    reduced_image_index_df = image_index_df[required_cols].copy()
    x_position_index = sorted(image_index_df["PositionX@um"].unique())
    y_position_index = sorted(image_index_df["PositionY@um"].unique())

    reduced_image_index_df["index_x"] = reduced_image_index_df["PositionX@um"].apply(x_position_index.index)
    reduced_image_index_df["index_y"] = reduced_image_index_df["PositionY@um"].apply(y_position_index.index)
    reduced_image_index_df["position_x"] = reduced_image_index_df["index_x"] * reduced_image_index_df["ImageSizeX"]
    reduced_image_index_df["position_y"] = reduced_image_index_df["index_y"] * reduced_image_index_df["ImageSizeY"]
    reduced_image_index_df = reduced_image_index_df.sort_values(["Plane", "Channel", "index_x", "index_y"])

    tiff_tag_lookup = invert_flat_dict(TAGS)
    width_tag = tiff_tag_lookup["ImageWidth"][0]
    height_tag = tiff_tag_lookup["ImageLength"][0]

    out_format = os.path.join(out_dir, out_format)
    for well, well_df in tqdm(reduced_image_index_df.groupby("WellName"), position=0):
        plane_stack = []
        tiffinfo = None
        for plane, plane_df in tqdm(well_df.sort_values("Plane").groupby("Plane"), position=1, leave=False):
            channel_stack = []
            for channel, channel_df in tqdm(plane_df.sort_values("Channel").groupby("Channel"), position=1, leave=False):
                all_images = [TiffImageFile(fp=fp) for fp in tqdm(channel_df["file_path"], position=2, leave=False)]
                transposed_images = [tiff_img.transpose(Image.FLIP_TOP_BOTTOM) for tiff_img in all_images]

                tiffinfo = dict(all_images[-1].tag)
                # we assume that all registered images have the same metadata
                # here we only look at the first image of the last channel to grab the metadata

                if correct_basic:
                    from basicpy import BaSiC
                    basic = BaSiC(**default_basic_kwargs)  # default is 10 iterations
                    stacked_images = np.stack(transposed_images, axis=0)
                    basic.fit(stacked_images)
                    images_transformed = basic.transform(stacked_images)

                    images_transformed = images_transformed.astype(stacked_images.dtype)  # to restore original type
                    transposed_images = [Image.fromarray(img, all_images[0].mode) for img in
                                         images_transformed]  # to restore original Image mode

                all_positions = channel_df[["position_x", "position_y", ]].values
                full_channel_image = merge_images(*transposed_images, positions=[tuple(xy) for xy in all_positions],
                                                  pad_color=0)
                channel_stack.append(np.array(full_channel_image))

            plane_image = np.dstack(channel_stack)
            plane_stack.append(plane_image)

        h, w, _c = plane_stack[0].shape
        tiffinfo[width_tag], tiffinfo[height_tag] = w, h

        tiffinfo[width_tag] = plane_stack[0].width
        tiffinfo[height_tag] = plane_stack[0].height

        imwrite(out_format.format(well), plane_stack, imagej=True, metadata=tiffinfo)
