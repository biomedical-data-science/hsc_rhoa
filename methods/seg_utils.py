from glob import glob
import pandas as pd
import numpy as np
import napari
import os

import matplotlib.pyplot as plt
from skimage import segmentation, measure
from skimage.restoration import denoise_tv_chambolle
from skimage.io import imread
from skimage.feature import peak_local_max
import scipy.ndimage as ndi

from methods import hsc_spatial_stats as hsc


def czi_image_preprocessing(
    czi_path,
    res,
    resize=True,
):
    """
    Preprocesses a CZI image, including denoising, resizing, segmentation,
    normalization, and optional plotting.

    Args:
        czi_path (str): Path to the .czi image file.
        channel (str): Name of the channel in the .czi file to be used.
        plot_path (str): Path for saving optional plots (if provided).

    Returns:
        img (ndarray): Preprocessed image data.
        nuc_mask (ndarray): Segmented nucleus mask.
        metadata_dic (dict): Dictionary containing image metadata.
    """
    # Load CZI image and metadata
    czi_img, metadata_dic = hsc.load_czi_image(czi_path)
    print(czi_img.shape)

    # Check if image loading was successful
    if not isinstance(czi_img, np.ndarray):
        return None, None, None

    # Denoise using the total variation in the 3D image
    img = denoise_tv_chambolle(czi_img, weight=0.01)

    # Resize image with isotropic interpolation
    if resize:
        img = hsc.resize_image(img, metadata_dic["original_res"], new_res=res)
        if img is None:
            return None, None, None

    # Get segmentation mask of the nucleus
    nuc_mask = hsc.get_nucleus_mask(img, res)

    # Check if a valid mask was obtained
    if len(np.unique(nuc_mask)) == 1:
        print("No mask found - skipping")
        return None, None

    # Remove zero dimensions from the image and mask
    img, nuc_mask = hsc.trim_zeros(img, nuc_mask)

    # Pad the image and mask to a desired shape (centering the object)
    img, nuc_mask = hsc.pad_center([img, nuc_mask])

    return img, nuc_mask


def visualize_comparison(im, manual_annot, otsu_seg):
    """
    Visualize original image with both segmentations for comparison
    """
    viewer = napari.Viewer()

    # Add layers
    viewer.add_image(im, name="Original", colormap="gray")
    viewer.add_labels(manual_annot, name="Manual Segmentation", opacity=0.7)
    viewer.add_labels(otsu_seg, name="Otsu Segmentation", opacity=0.5)

    print("Comparison view loaded. Close viewer when done examining.")
    viewer.show(block=False)


def get_nuc_filename(input, folder, cond):

    # Extract batch ID and nucleus ID from input path
    batch_id = os.path.basename(os.path.dirname(input))
    nuc_id = os.path.basename(input)[:-4]
    nuc_id = nuc_id.replace("Image ", "nuc_")

    # Create the filename based on batch and nucleus ID
    filename = f"{folder}/{cond}/{batch_id}_{nuc_id}"

    return filename


def compute_segmentation_metrics(manual, otsu):

    # Calculate basic quantities
    true_pos = np.sum(manual * otsu)
    false_pos = np.sum((1 - manual) * otsu)
    false_neg = np.sum(manual * (1 - otsu))
    true_neg = np.sum((1 - manual) * (1 - otsu))

    # Calculate metrics
    precision = true_pos / (true_pos + false_pos)
    recall = true_pos / (true_pos + false_neg)
    dice_score = (2.0 * true_pos) / (2.0 * true_pos + false_pos + false_neg)
    jaccard = true_pos / (true_pos + false_pos + false_neg)
    specificity = true_neg / (true_neg + false_pos)
    accuracy = (true_pos + true_neg) / (
        true_pos + true_neg + false_pos + false_neg
    )

    return pd.DataFrame(
        {
            "Dice Score": [dice_score],
            "Precision": [precision],
            "Recall": [recall],
            "Jaccard Score": [jaccard],
            "Specificity": [specificity],
            "Accuracy": [accuracy],
        }
    )


def calculate_segmentation_metrics_3D(conditions, folder, res):

    metrics_df = pd.DataFrame()

    # For each condition
    for cond in conditions:

        # Get a list of image files (CZI format) within the condition directory
        img_list = glob(f"{folder}/{cond}/*.czi", recursive=True)

        # For each image within this condition
        for img_path in img_list:
            print(img_path)

            annot_path = img_path.replace(".czi", ".tif")

            im, nuc_mask = czi_image_preprocessing(img_path, res)
            n_ims_out = int(im.shape[0] * 0.2)
            n_down = n_ims_out
            n_up = im.shape[0] - n_ims_out
            print(n_ims_out)

            manual_annot = imread(annot_path)

            manual_binary = (manual_annot > 0).astype(np.uint8).flatten()
            otsu_binary = (
                (nuc_mask[n_down:n_up] > 0).astype(np.uint8).flatten()
            )
            print(manual_binary.shape, otsu_binary.shape)
            this_metrics = compute_segmentation_metrics(
                manual_binary, otsu_binary
            )
            this_metrics["Condition"] = cond
            this_metrics["Nucleus"] = img_path

            metrics_df = pd.concat(
                [metrics_df, this_metrics],
                ignore_index=True,
            )

            # visualize_comparison(
            #    im[n_down:n_up], manual_annot, nuc_mask[n_down:n_up]
            # )

    return metrics_df


def calculate_segmentation_metrics_2D(conditions, folder, res):

    metrics_df = pd.DataFrame()

    for cond in conditions:
        img_list = glob(f"{folder}/{cond}/*.czi", recursive=True)

        for img_path in img_list:
            print(img_path)
            annot_path = img_path.replace(".czi", ".tif")

            im, nuc_mask = czi_image_preprocessing(img_path, res)
            manual_annot = imread(annot_path)

            n_ims_out = int(im.shape[0] * 0.2)
            n_down = n_ims_out
            n_up = im.shape[0] - n_ims_out

            total_slices = n_up - n_down

            for slice_idx in range(n_down, n_up):

                manual_slice = manual_annot[slice_idx - n_down]
                nuc_slice = nuc_mask[slice_idx]

                manual_binary = (manual_slice > 0).astype(np.uint8).flatten()
                otsu_binary = (nuc_slice > 0).astype(np.uint8).flatten()

                this_metrics = compute_segmentation_metrics(
                    manual_binary, otsu_binary
                )

                # Add comprehensive metadata
                this_metrics["Condition"] = cond
                this_metrics["Nucleus"] = img_path
                this_metrics["Slice Index"] = slice_idx
                this_metrics["N slices"] = total_slices

                metrics_df = pd.concat(
                    [metrics_df, this_metrics],
                    ignore_index=True,
                )

    return metrics_df


def plot_masks_contour(im, mask1, mask2):

    _, ax = plt.subplots(figsize=(10, 6))
    plot = ax.imshow(im, cmap="viridis")

    for mask, col in zip([mask1, mask2], ["whitesmoke", "red"]):

        mask_border = segmentation.find_boundaries(mask, mode="inner").astype(
            np.uint8
        )
        coords = np.column_stack(np.where(mask_border > 0))

        ax.plot(
            coords[:, 1],
            coords[:, 0],
            linestyle="None",
            marker="s",
            color=col,
        )

    # Draw the resolution marker line depicting 1 micrometer
    for i, num in enumerate(range(3, 13)):
        ax.plot(num, 2, linestyle="None", marker="s", color="whitesmoke")

    cbar = plt.colorbar(plot, ax=ax, aspect=5)
    cbar.ax.tick_params(labelsize=40)

    return ax


def update_labels(df):
    """
    Fix the label for each HSC biological condition in the dataframe from the
    folder ID name to new names for plotting
    """

    labels_dict = {
        "young": "Young",
        "aged": "Aged",
        "Aged_treated_RhoAi": "Aged + Ri",
    }

    for old, new in labels_dict.items():
        df["Condition"] = [c.replace(old, new) for c in df["Condition"]]

    return df


def plot_mask_contour(mask, ax, color="whitesmoke"):

    mask_border = segmentation.find_boundaries(mask, mode="inner").astype(
        np.uint8
    )
    coords = np.column_stack(np.where(mask_border > 0))
    ax.plot(
        coords[:, 1],
        coords[:, 0],
        linestyle="None",
        marker="s",
        color=color,
    )

    return ax


def segment_condensates(im, mask, cluster_thresh=1.8, mode="relative"):

    # Create binary mask with blobs
    nucleus_mean = np.mean(im[mask > 0.5])
    if mode == "relative":
        def_thres = nucleus_mean * cluster_thresh
    elif mode == "absolute":
        def_thres = cluster_thresh

    blob_mask = im > def_thres

    # Find local maximas and create mask with these dots
    distance = ndi.distance_transform_edt(blob_mask)
    local_max_coords = peak_local_max(distance, min_distance=5)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = measure.label(local_max_mask)

    # Segment with Watershed algorithm
    segmented = segmentation.watershed(-distance, markers, mask=blob_mask)

    return segmented
