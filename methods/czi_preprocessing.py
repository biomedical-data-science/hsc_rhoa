import os
import gc
import zarr
import random
from glob import glob
import numpy as np
import pandas as pd
import skimage as ski
from matplotlib import pyplot as plt

from scipy.ndimage import binary_fill_holes
from aicsimageio import AICSImage


random.seed(2023)


class CZIPreprocessing:
    """
    A class for preprocessing CZI microscopy image data with customizable parameters.

    Input Parameters:
        input_dir (str): The directory containing input CZI files.
        output_dir (str): The directory where preprocessed data will be saved.
        conditions (list): A list of conditions or samples to process.
        resolution (float): A float representing the desired isotropic resolution for resizing.
        normalization (str): The normalization method to apply (e.g. 'minmax', 'zscore').
        channel (str): The specific channel to process.
        resize (bool): Whether to perform resizing along all dimensions (default: True).
        resize_z (bool): Whether to perform z-axis resizing (default: True).
        outliers (list): A list containing previously detected outlier filepaths (optional).
        override (bool): Whether to save a new version of the file if it already exists.
    """

    def __init__(
        self,
        input_dir,
        output_dir,
        conditions,
        resolution,
        normalization,
        channel,
        resize=True,
        resize_z=True,
        outliers=None,
        override=False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.conditions = conditions
        self.outliers = outliers
        self.res = resolution
        self.norm = normalization
        self.channel = channel
        self.resize = resize
        self.resize_z = resize_z
        self.override = override

    def save_as_3D_array(self, extension="zip"):
        """
        Save CZI images as 3D arrays with optional zip or npz compression.

        Parameters:
            extension (str, optional): Output file extension ('zip' or 'npz'). Default is 'zip'.

        Returns:
            list: List of output directories where the processed data is saved.
        """
        # Create output directory if it does not exist already
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        output_dir_list = []  # List to store output directories

        # For each condition
        for cond in self.conditions:
            # Create condition folder if it does not exist already
            class_dir = f"{self.output_dir}/{cond}"
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

            # Get a list of image files (CZI format) within the condition directory
            img_list = glob(
                f"{self.input_dir}/{cond}/**/*.czi", recursive=True
            )

            # For each image within this condition
            for i, img_path in enumerate(img_list, 1):
                out_path = self._get_nuc_filename(img_path, class_dir)

                # If this cell was already processed and override mode is not enabled - skip
                if not self.override:
                    if os.path.exists(
                        f"{out_path}_zarr.zip"
                    ) or os.path.exists(f"{out_path}.npz"):
                        continue

                # Skip outliers
                if self.outliers is not None:
                    if img_path in self.outliers:
                        print("Outlier identified - skipping")
                        continue

                # Print information to the user
                print(img_path)
                print("Image Number : " + str(i))

                # Preprocess the CZI image and obtain image data, nucleus mask, and metadata
                img, nuc_mask, metadata = self.czi_image_preprocessing(
                    img_path, plot_path=out_path
                )

                # If the image could not be extracted, skip to the next image
                if img is None:
                    print(
                        "Image could not be extracted from the CZI file - skipping"
                    )
                    continue

                # Update metadata with relevant information
                metadata["czi_path"] = img_path
                metadata["condition"] = cond
                metadata["normalization"] = self.norm
                metadata["resolution"] = self.res
                metadata["resized"] = self.resize
                output_dir_list.extend(out_path)

                # If the chosen extension is 'zip', save data as zarr format
                if extension == "zip":
                    zarr.save(
                        f"{out_path}.{extension}",
                        img=img.astype(np.float32),
                        nuc_mask=nuc_mask.astype(float),
                    )
                    # Save metadata as a CSV file
                    pd.Series(metadata).to_csv(f"{out_path}_metadata.csv")

                # If the chosen extension is 'npz', save data as compressed numpy format
                elif extension == "npz":
                    np.savez_compressed(
                        out_path,
                        img=img.astype(np.float32),
                        nuc_mask=nuc_mask.astype(float),
                        metadata=metadata,
                    )

                gc.collect()  # Perform garbage collection to free up memory

        return output_dir_list

    def _get_channel_image_and_metadata(
        self, czi_img, czi_filepath, channel_idx
    ):
        """
        Extract an image and metadata for a specific channel from a CZI file.

        Args:
            self: The instance of the class.
            czi_img: The CZI image object.
            czi_filepath (str): The file path of the CZI image.
            channel_idx (int): Index of the channel to extract.

        Returns:
            img (numpy.ndarray): The channel image data.
            metadata_dic (dict): Metadata dictionary containing information about the image.
        """
        # Extract the channel image data
        img = czi_img.get_image_data("ZXY", C=channel_idx)

        # Extract batch ID and nucleus ID from file paths
        batch_id = os.path.basename(os.path.dirname(czi_filepath))
        nuc_id = os.path.basename(czi_filepath)[:-4]
        nuc_id = nuc_id.replace("Image ", "nuc_")

        # Fix the Dimension object to make it a serialisable dictionary for JSON
        pix_sizes = czi_img.physical_pixel_sizes
        original_res = {"Z": pix_sizes.Z, "X": pix_sizes.X, "Y": pix_sizes.Y}
        original_dims = {k: v for k, v in czi_img.dims.items()}

        metadata_dic = {
            "original_res": original_res,
            "original_dims": original_dims,
            "original_num_channels": czi_img.dims.C,
            "original_channel_names": czi_img.channel_names,
            "original_sigma_noise": ski.restoration.estimate_sigma(img),
            "original_intensity_sum": img.sum(),
            "original_intensity_mean": img.mean(),
            "original_intensity_max": img.max(),
            "original_intensity_min": img.min(),
            "nuc_id": nuc_id,
            "batch_id": batch_id,
        }

        return img, metadata_dic

    def _load_czi_image(self, czi_filepath):
        """
        Load a CZI image file and extract a specific channel along with metadata.

        Args:
            czi_filepath (str): The file path of the CZI image.

        Returns:
            img (numpy.ndarray): The channel image data.
            metadata_dic (dict): Metadata dictionary containing information about the image.
        """
        # Load the CZI image using AICSImage
        czi_img = AICSImage(czi_filepath)

        # Check if the CZI image has channel names information
        if "get_channel_names" in dir(czi_img):
            channel_names_list = czi_img.get_channel_names()
        elif "channel_names":
            channel_names_list = czi_img.channel_names
        else:
            print(
                "WARNING: Skiping image. No AICSimage compatible channel detected"
            )
            return None, None

        # Find the index of the specified channel
        channel = [s for s in channel_names_list if self.channel in s]
        if not channel:
            print(f"WARNING, Skiping image. No {channel} channel.")
            return None, None

        channel_idx = channel_names_list.index(channel[0])

        # Extract the channel image and metadata using the index
        return self._get_channel_image_and_metadata(
            czi_img, czi_filepath, channel_idx
        )

    def _resize_image(self, img, pixel_sizes):
        """
        Resize an image while preserving its aspect ratio, achieving the same resolution
        for each of the X, Y and Z dimensions.

        Args:
            self: The instance of the class.
            img (numpy.ndarray): The input image to be resized.
            pixel_sizes (dict): Dictionary containing pixel sizes for Z, X, and Y axes.

        Returns:
            img (numpy.ndarray): The resized image.
        """
        # Calculate the new shape based on desired pixel sizes and current resolution
        new_shape = (
            np.round(img.shape[0] * pixel_sizes["Z"] / self.res),
            np.round(img.shape[1] * pixel_sizes["X"] / self.res),
            np.round(img.shape[2] * pixel_sizes["Y"] / self.res),
        )

        # Check that the number of pixels is the same in X and Y
        for i, _ in enumerate(new_shape[1:], 1):
            if new_shape[i] > img.shape[i]:
                print("Skipping due to different X and Y resolution")
                return None

        # Resize the image using scikit-image, preserving the image range and applying anti-aliasing
        # About anti-aliasing:
        # Whether to apply a Gaussian filter to smooth the image prior to
        # downsampling. It is crucial to filter when downsampling the image to
        # avoid aliasing artifacts. If not specified, it is set to True when
        # downsampling an image whose data type is not bool.
        if not self.resize_z:
            new_shape[0] = img.shape[0]

        img = ski.transform.resize(
            img,
            new_shape,
            preserve_range=True,
            anti_aliasing=True,
        )

        print("Rescaling image. New Image Size: ", img.shape)
        return img

    def _zscore_normalization(self, img, mask):
        """Perform Z-score normalization (Standardization) in the
        image region inside the supplied nuclear mask"""
        roi = img[mask.astype(bool)]
        img = (img - roi.mean()) / roi.std()

        return img

    def _mean_normalization(self, img, mask):
        """Perform Mean normalization in the image region inside
        the supplied nuclear mask"""
        roi = img[mask.astype(bool)]
        img -= roi.mean()
        return img

    def _pad_center(self, array_list, padding=(3, 3)):
        """
        Pad each array in array_list with zeros to center it within a specified padding.

        Args:
            array_list (list of numpy.ndarray): List of arrays to be padded.
            padding (tuple): Number of pixels to pad for each dimension.

        Returns:
            list of numpy.ndarray: List of padded arrays.
        """
        # Determine the number of dimensions in the arrays
        n_dims = len(array_list[0].shape)

        # Create a pad width list for each dimension using the specified padding
        pad_width = [padding for n in range(n_dims)]

        # Pad each array in array_list with zeros using the specified padding
        return [np.pad(a, pad_width=pad_width) for a in array_list]

    def _trim_zeros(self, img, mask):
        """Returns a trimmed view of an n-D array excluding regions containing
        only zeros"""

        # Find non-zero indices in the mask for each dimension
        non_zero = np.nonzero(mask)

        # Create slices that define the trimmed region
        slices = tuple(slice(idx.min(), idx.max() + 1) for idx in non_zero)

        # Return the trimmed image and mask using the defined slices
        return img[slices], mask[slices]

    def _get_nucleus_mask(self, img, tolerance=0.2):
        """Batch calling different segmentation steps
        INPUT:
            img: image numpy 3D array
            res: image resolution
        """
        sigma = 0.1 / self.res

        smooth_img = ski.filters.gaussian(img, sigma=sigma)
        otsu_thres = ski.filters.threshold_otsu(smooth_img)
        lower_thres = otsu_thres * (1 - tolerance)
        upper_thres = otsu_thres * (1 + tolerance)

        # applies histerisis thresholding of TOLERANCE
        mask = ski.filters.apply_hysteresis_threshold(
            smooth_img, lower_thres, upper_thres
        )

        # fills holes
        mask = ski.morphology.binary_closing(mask, ski.morphology.ball(5))
        mask = binary_fill_holes(mask, ski.morphology.ball(5))

        return mask

    def _plot_seg(self, img, mask, plane, out_path, contour=True, plot=False):
        """
        Plots an input image with contoured segmentation in a matrix of subplots
        with different slices in the indicated plane.

        Args:
            img (numpy.ndarray): Input image as a 3D numpy array.
            mask (numpy.ndarray): Segmented image as a binary 3D mask.
            plane (str): Plane to plot/slice (e.g., 'XY', 'ZX', 'ZY').
            out_path (str): Filename for the resulting image.
            contour (bool): Whether to plot the contour of the nucleus mask.
            plot (bool): Whether to plot the image on the terminal.

        Returns:
            None
        """
        # Constants for subplot layout
        n_bs = 3  # Number of rows per subplot
        n_total = 15  # Total number of slices
        n_cols = int(np.ceil(n_total / n_bs))  # Number of columns
        fig, axes = plt.subplots(3, n_cols, figsize=(15, 10))

        # For each of the created axes and 2D slides, plot depending on the chosen plane
        for i, ax in enumerate(axes.flat):
            if plane == "XY":
                slice_img = img[
                    i * int(np.floor(img.shape[0] / n_total)), :, :
                ]
                slice_mask = mask[
                    i * int(np.floor(img.shape[0] / n_total)), :, :
                ]
            elif plane == "ZX":
                slice_img = img[
                    :, :, i * int(np.floor(img.shape[2] / n_total))
                ]
                slice_mask = mask[
                    :, :, i * int(np.floor(img.shape[2] / n_total))
                ]
            elif plane == "ZY":
                slice_img = img[
                    :, i * int(np.floor(img.shape[1] / n_total)), :
                ]
                slice_mask = mask[
                    :, i * int(np.floor(mask.shape[1] / n_total)), :
                ]

            # Display the image and optional contour
            ax.imshow(slice_img, vmax=np.max(img.flatten()) * 0.7)
            if contour:
                ax.contour(slice_mask)
            ax.set_title(f"PLANE : {plane}")

        # Save the resulting image and close
        plt.savefig(out_path)
        plt.close()

        # Optionally, display the plot on the terminal
        if plot:
            plt.show()

    def _get_nuc_filename(self, input, output, suffix="XY", ext=".png"):
        """
        Generates an output filename for .png inspection images based on input.

        Args:
            input (str): Input filename of .czi image.
            output (str): Working output folder.
            suffix (str): Suffix to be added before the file extension.
            ext (str): File extension for the output filename.

        Returns:
            str: The generated output filename.
        """

        # Extract batch ID and nucleus ID from input path
        batch_id = os.path.basename(os.path.dirname(input))
        nuc_id = os.path.basename(input)[:-4]
        nuc_id = nuc_id.replace("Image ", "nuc_")

        # Create the filename based on batch and nucleus ID
        filename = f"{output}/{batch_id}_{nuc_id}"

        return filename

    def czi_image_preprocessing(self, czi_path, plot_path=None):
        """
        Preprocesses a CZI image, including denoising, resizing, segmentation,
        normalization, and optional plotting.

        Args:
            self: The instance of the class.
            czi_path (str): Path to the .czi image file.
            channel (str): Name of the channel in the .czi file to be used.
            plot_path (str): Path for saving optional plots (if provided).

        Returns:
            img (ndarray): Preprocessed image data.
            nuc_mask (ndarray): Segmented nucleus mask.
            metadata_dic (dict): Dictionary containing image metadata.
        """
        # Load CZI image and metadata
        czi_img, metadata_dic = self._load_czi_image(czi_path)

        # Check if image loading was successful
        if not isinstance(czi_img, np.ndarray):
            return None, None, None

        # Denoise using the total variation in the 3D image
        img = ski.restoration.denoise_tv_chambolle(czi_img, weight=0.01)

        # Resize image with isotropic interpolation
        if self.resize:
            img = self._resize_image(img, metadata_dic["original_res"])
            if img is None:
                return None, None, None

        # Get segmentation mask of the nucleus
        nuc_mask = self._get_nucleus_mask(img)

        # Check if a valid mask was obtained
        if len(np.unique(nuc_mask)) == 1:
            print("No mask found - skipping")
            return None, None, None

        # Remove zero dimensions from the image and mask
        img, nuc_mask = self._trim_zeros(img, nuc_mask)

        # Pad the image and mask to a desired shape (centering the object)
        img, nuc_mask = self._pad_center([img, nuc_mask])

        # Eliminate residual intensities outside the nucleus mask
        img = img * nuc_mask

        # TV-chambolle and resizing make the intensity values very small.
        # Rescale intensity values to the 0-1 interval
        min = np.min(img[nuc_mask.astype(bool)])
        max = np.max(img[nuc_mask.astype(bool)])
        img = (img - min) / (max - min)

        # Normalize the image using different methods (z-score or mean)
        if self.norm == "z_score":
            img = self._zscore_normalization(img, mask=nuc_mask)
        elif self.norm == "mean":
            img = self._mean_normalization(img, mask=nuc_mask)

        print(img.shape)

        # Optional plotting of segmented images
        if plot_path:
            for suffix in ["XY", "ZY"]:
                out_path = f"{plot_path}_{suffix}.png"
                self._plot_seg(img, nuc_mask, suffix, out_path, contour=True)

        return img, nuc_mask, metadata_dic
