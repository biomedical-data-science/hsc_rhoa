import os
import numpy as np
import pandas as pd
import skimage as ski
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.ticker as ticker
from scipy.stats import mannwhitneyu


properties_2d = [
    "area",
    "bbox_area",
    "convex_area",
    "bbox",
    "centroid",
    "major_axis_length",
    "minor_axis_length",
    "mean_intensity",
    "min_intensity",
    "max_intensity",
    "label",
    "solidity",
    "orientation",
    "perimeter",
]

properties_3d = [
    "area",
    "bbox_area",
    "convex_area",
    "equivalent_diameter",
    "feret_diameter_max",
    "filled_area",
    "bbox",
    "centroid",
    "major_axis_length",
    "minor_axis_length",
    "mean_intensity",
    "min_intensity",
    "max_intensity",
    "label",
    "solidity",
]

properties_3d_DIR = [
    "area",
    "bbox_area",
    "convex_area",
    "equivalent_diameter",
    "filled_area",
    "bbox",
    "centroid",
    "major_axis_length",
    "mean_intensity",
    "min_intensity",
    "max_intensity",
    "label",
    "solidity",
    # It results in an error
    # "feret_diameter_max",
    # "minor_axis_length",
]


def get_channel_image_and_metadata(czi_img, czi_filepath, channel_idx):
    """
    Extract an image and metadata for a specific channel from a CZI file.

    Args:
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


def load_czi_image(czi_filepath, channel_name="Ch1-T2"):
    """
    Load a CZI image file and extract a specific channel along with metadata.

    Args:
        czi_filepath (str): The file path of the CZI image.
        channel_name (str): The name of the channel to extract (default is "Ch1").

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
    channel = [s for s in channel_names_list if channel_name in s]
    if not channel:
        print(f"WARNING, Skiping image. No {channel} channel.")
        return None, None

    channel_idx = channel_names_list.index(channel[0])

    # Extract the channel image and metadata using the index
    return get_channel_image_and_metadata(czi_img, czi_filepath, channel_idx)


def resize_image(img, pixel_sizes, new_res=0.1, resize_z=True):
    """Given an image and the original pixel sizes, return the image resized
    to the new resolution"""
    new_shape = (
        np.round(img.shape[0] * pixel_sizes["Z"] / new_res),
        np.round(img.shape[1] * pixel_sizes["X"] / new_res),
        np.round(img.shape[2] * pixel_sizes["Y"] / new_res),
    )

    # Check that the number of pixels won't be bigger in X and Y
    for i, _ in enumerate(new_shape[1:], 1):
        if new_shape[i] > img.shape[i]:
            print("Skipping due to outlier X and Y resolution")
            return None

    # Resize the image using scikit-image, preserving the image range and applying anti-aliasing
    # About anti-aliasing:
    # Whether to apply a Gaussian filter to smooth the image prior to
    # downsampling. It is crucial to filter when downsampling the image to
    # avoid aliasing artifacts. If not specified, it is set to True when
    # downsampling an image whose data type is not bool.
    if not resize_z:
        new_shape[0] = img.shape[0]

    img = ski.transform.resize(
        img,
        new_shape,
        preserve_range=True,
        anti_aliasing=True,
    )

    print("Rescaling image. New Image Size: ", img.shape)

    return img


def zscore_normalization(im, mask):
    """Perform Z-score normalization (Standardization)"""
    roi = im[mask.astype(bool)]

    im -= roi.mean()
    im /= roi.std()
    return im


def mean_normalization(im, mask):
    """Perform mean normalization"""
    roi = im[mask.astype(bool)]
    im -= roi.mean()
    return im


def pad_center(array_list, padding=(3, 3)):
    """For each of the arrays in array_list, pad the array with 0s for the
    number of pixels specified in padding"""
    pad_width = [padding for n in range(len(array_list[0].shape))]
    return [np.pad(a, pad_width=pad_width) for a in array_list]


def from_zxy_to_xyz(array_list):
    """Transpose the image dimensions from ZXY to XYZ coordinates"""
    return [np.transpose(a, axes=[1, 2, 0]) for a in array_list]


def trim_zeros(img, mask):
    """Returns a trimmed view of an n-D array excluding regions containing
    only zeros"""

    # Find non-zero indices in the mask for each dimension
    non_zero = np.nonzero(mask)

    # Create slices that define the trimmed region
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in non_zero)

    # Return the trimmed image and mask using the defined slices
    return img[slices], mask[slices]


def get_nucleus_mask(img, res, tolerance=0.2):
    """Batch calling different segmentation steps
    INPUT:
        img: image numpy 3D array
        res: image resolution
    """
    sigma = 0.1 / res

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


def get_output_filename(
    input_path, output_path, suffix="XY", extension=".png"
):
    """Generates output filename for .png inspection images
    INPUT:
          input_path: input filename of .czi image
          output_path: working output folder
          suffix: suffix to be added before file extension
    """
    filename = (
        os.path.splitext(os.path.basename(input_path))[0]
        + "_"
        + suffix
        + extension
    )

    return os.path.join(output_path, filename.replace(" ", "_"))


def get_output_filename_with_subfolder(
    input_path, output_path, suffix="XY", extension=".png"
):
    """generates output filename for .png inspection images adding an
    intermediate subfolder
    INPUT:
          input_path: input filename of .czi image
          output_path: working output folder
          suffix: suffix to be added before file extension
    """

    filename = (
        os.path.splitext(os.path.basename(input_path))[0]
        + "_"
        + os.path.basename(os.path.dirname(input_path))
        + "_"
        + suffix
        + extension
    )
    return os.path.join(output_path, filename.replace(" ", "_"))


def czi_image_preprocessing(
    czi_path, resize=True, norm="None", plot_path=None
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
    czi_img, metadata_dic = load_czi_image(czi_path)

    # Check if image loading was successful
    if not isinstance(czi_img, np.ndarray):
        return None, None, None

    # Denoise using the total variation in the 3D image
    img = ski.restoration.denoise_tv_chambolle(czi_img, weight=0.01)

    # Resize image with isotropic interpolation
    if resize:
        img = resize_image(img, metadata_dic["original_res"])
        if img is None:
            return None, None, None

    # Get segmentation mask of the nucleus
    nuc_mask = get_nucleus_mask(img)

    # Check if a valid mask was obtained
    if len(np.unique(nuc_mask)) == 1:
        print("No mask found - skipping")
        return None, None, None

    # Remove zero dimensions from the image and mask
    img, nuc_mask = trim_zeros(img, nuc_mask)

    # Pad the image and mask to a desired shape (centering the object)
    img, nuc_mask = pad_center([img, nuc_mask])

    # Eliminate residual intensities outside the nucleus mask
    img = img * nuc_mask

    # TV-chambolle and resizing make the intensity values very small.
    # Rescale intensity values to the 0-1 interval
    min = np.min(img[nuc_mask.astype(bool)])
    max = np.max(img[nuc_mask.astype(bool)])
    img = (img - min) / (max - min)

    # Normalize the image using different methods (z-score or mean)
    if norm == "z_score":
        img = zscore_normalization(img, mask=nuc_mask)
    elif norm == "mean":
        img = mean_normalization(img, mask=nuc_mask)

    print(img.shape)

    # Optional plotting of segmented images
    if plot_path:
        for suffix in ["XY", "ZY"]:
            out_path = f"{plot_path}_{suffix}.png"
            plot_seg(img, nuc_mask, suffix, out_path, contour=True)

    return img, nuc_mask, metadata_dic


def get_morphometrics_from_largest_DIR(mask, im):
    """Get morphometric b with volume and morphometric values for the larger
    DIR in the segmentation"""

    info_table = pd.DataFrame(
        ski.measure.regionprops_table(
            mask.astype(np.uint8),
            im,
            properties=[
                "area",
                "mean_intensity",
                "solidity",
                "convex_area",
                "centroid",
                "inertia_tensor_eigvals",
                "major_axis_length",
                "minor_axis_length",
                "equivalent_diameter",
                "weighted_centroid",
            ],
        )
    )

    (b,) = np.where(info_table.area == max(info_table.area))
    new_entry = info_table.iloc[b,]

    nlabels = info_table.shape[0]  # number of labels
    new_entry["nlabels"] = nlabels

    return new_entry


def get_surface_area(mask, sigma=2):
    """Reconstructs mesh surface on binary 3d mask and estimates a surface
    measure
    INPUT
          mask: segmentated nucleus as binary numpy array
          sigma: smoothing of border prior to the marching cubes algorithm
          to avoid overfitting pixelized border
    """

    mask_smooth = ski.filters.gaussian(mask, sigma=sigma)
    verts, faces, normals, values = ski.measure.marching_cubes(
        mask_smooth, level=None, step_size=2
    )

    return ski.measure.mesh_surface_area(verts, faces)


def plot_intensity_profile_3D(
    im, mask, dist2bord, res, max_dist=20, thickness=1.5
):
    """Method to plot an Average Mean vs. Distance to edge curve for a single
    nucleus with interquartile intervals
    INPUT:
          img: image numpy 3D array
          mask: segmentated nucleus as binary numpy array
          dist2bord: numpy array containing all the closest distances to
          the border for each voxel (euclidian distance transform)
          res: image resolution
          max_dist: maximum distance in pixels to measure
          thickness: band thickness in pixels to measure
    """
    q1_allint = np.quantile(im[mask > 0.1], 0.25)
    q3_allint = np.quantile(im[mask > 0.1], 0.75)

    mean_int, nvox, q1_int, q3_int = [
        np.ones(max_dist) * np.NaN for i in range(4)
    ]

    for lag, dist in enumerate(range(max_dist)):
        coords = tuple(
            np.where(
                (dist2bord < (dist + thickness))
                & (dist2bord >= dist)
                & (mask > 0.1)
            )
        )
        if coords[0].shape[0] == 0:
            continue

        mean_int[lag] = np.mean(im[coords])
        q1_int[lag] = np.quantile(im[coords], 0.25)
        q3_int[lag] = np.quantile(im[coords], 0.75)
        nvox[lag] = len(coords[0])

    distances = np.asarray(range(max_dist)) * res

    plt.errorbar(
        x=distances,
        y=mean_int,
        fmt="o",
        linestyle="-",
        yerr=[mean_int - q1_int, q3_int - mean_int],
    )
    plt.axhline(y=q1_allint, color="r", linestyle="--", alpha=0.5)
    plt.axhline(y=q3_allint, color="r", linestyle="--", alpha=0.5)
    plt.title("Mean DAPI Intensity vs. Distance to Border")
    plt.show()


def get_intensity_in_distance(im, mask, dist2bord, distance_range=(0, 10)):
    """Get average pixel intensity value in the given distance range"""
    min_dist, max_dist = distance_range

    coords = tuple(
        np.where(
            (dist2bord < max_dist) & (dist2bord >= min_dist) & (mask > 0.1)
        )
    )

    # If that distance range is not found on the image, NaN is returned
    return np.mean(im[coords])


def get_intensity_by_distance(
    im, mask, dist2bord, res, max_dist=20, thickness=1.5
):
    """Get average pixel intensity value by each pixel distance up until the
    maximum distance specified"""
    int_df = pd.DataFrame()

    for dist in range(max_dist):
        mean_int = get_intensity_in_distance(
            im, mask, dist2bord, distance_range=(dist, (dist + thickness))
        )
        df = pd.DataFrame({"distance": dist * res, "mean_int": [mean_int]})
        # Delete from the dataframe those distances with NaN
        df = df.dropna()
        int_df = pd.concat([int_df, df])

    return int_df


def segment_condensates(im, mask, cluster_thresh=0.6, mode="absolute"):
    """Segment DAPI-Intense Regions (DIRs) using the specified clustering
    threshold
    INPUT:
        im: intensity image
        mask: segmentation mask
        cluster_thresh: threshold for the intensities to consider a DIR
        mode: whether use the cluster threshold as absolute over all the
        image pixel intensities or relative compared to the mean intensity
    """

    # Create binary mask with DIRs
    if mode == "relative":
        nucleus_mean = np.mean(im[mask > 0.5])
        def_thres = nucleus_mean * cluster_thresh
    elif mode == "absolute":
        def_thres = cluster_thresh

    DIR_mask = im > def_thres

    # Find local maximas and create mask with these dots
    distance = distance_transform_edt(DIR_mask)
    local_max_coords = ski.feature.peak_local_max(distance, min_distance=5)
    local_max_mask = np.zeros(distance.shape, dtype=bool)
    local_max_mask[tuple(local_max_coords.T)] = True
    markers = ski.measure.label(local_max_mask)

    # Segment with Watershed algorithm
    segmented = ski.segmentation.watershed(-distance, markers, mask=DIR_mask)

    return segmented


def get_distance_to_edge_by_DIR(df, dist2bord, DIRs_mask):
    """Get the distance to the nuclear edge by each DIR of a DIRs mask"""
    dims = range(len(DIRs_mask.shape))

    for b in df.index:
        # doesn't compute for non-included
        if not df.include[b]:
            continue

        # computes distance to edge
        centroid_coords = [
            np.round(df.loc[b, f"centroid-{d}"]).astype(np.uint8) for d in dims
        ]

        df.loc[b, "centroid_distance2border"] = dist2bord[
            tuple(centroid_coords)
        ]

        df.loc[b, "mean_distance2border"] = np.mean(
            dist2bord[DIRs_mask == (b + 1)].flatten()
        )

        df.loc[b, "min_distance2border"] = np.min(
            dist2bord[DIRs_mask == (b + 1)].flatten()
        )

    return df


def dist(p1, p2):
    """Calculate the euclidean distance between two points"""
    return np.sqrt(np.sum(np.square(np.array(p1) - np.array(p2))))


def calculate_ellipse_perimeter(a, b):
    """Calculate the ellipse of a parameter with major axis length a and minor
    axis length b"""
    # Formula: PI * ( 3*(a + b) - SQRT( (3*a + b) * (a + 3*b) ) )
    return np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))


def excess_of_perimeter_ellipse(obj):
    """Calculaten the Excess of Perimeter (EOP) of an object compared to an
    ellipse with the same major and minor axis lengths"""
    ellipse_p = calculate_ellipse_perimeter(
        obj.major_axis_length / 2, obj.minor_axis_length / 2
    )
    eop = (obj.perimeter - ellipse_p) / ellipse_p
    return eop


def get_slide(im, mask, mode="middle", plane="XY"):
    """Obtain a 2D slide out of a 3D numpy array"""

    assert len(im.shape) == 3, "input matrices are not 3D"
    assert im.shape == mask.shape, "image shape and mask shape are not equal"
    assert plane in ["XY", "ZX", "ZY"], "plane should be XY, ZX or ZY"

    if mode == "middle":
        # returns the middle slide
        if plane == "XY":
            mid_zslice = np.floor(im.shape[2] / 2)
            im, mask = im[:, :, mid_zslice], mask[:, :, mid_zslice]
        elif plane == "ZX":
            mid_zslice = np.floor(im.shape[1] / 2)
            im, mask = im[:, mid_zslice, :], mask[:, mid_zslice, :]
        elif plane == "ZY":
            mid_zslice = np.floor(im.shape[0] / 2)
            im, mask = im[mid_zslice, :, :], mask[mid_zslice, :, :]

    elif mode == "max":
        # return a slide with the maximum value found for each pixel
        if plane == "XY":
            im, mask = im.max(axis=2), mask.max(axis=2)
        elif plane == "ZX":
            im, mask = im.max(axis=1), mask.max(axis=1)
        elif plane == "ZY":
            im, mask = im.max(axis=0), mask.max(axis=0)

    elif mode == "largest":
        # returns the slide with the maximum area
        if plane == "XY":
            max_index = np.argmax(mask.sum(0).sum(0))
            im, mask = im[:, :, max_index], mask[:, :, max_index]
        elif plane == "ZX":
            max_index = np.argmax(mask.sum(0).sum(1))
            im, mask = im[:, max_index, :], mask[:, max_index, :]
        elif plane == "ZY":
            max_index = np.argmax(mask.sum(1).sum(1))
            im, mask = im[max_index, :, :], mask[max_index, :, :]

    return im, mask


def get_nuclear_metrics_3D(mask, im, res):
    """Extract features from a 3D nuclear image and mask"""
    label_mask = ski.measure.label(mask)

    df = pd.DataFrame(
        ski.measure.regionprops_table(
            label_mask, intensity_image=im, properties=properties_3d
        )
    )

    # turn pixel units into um units
    for m in [
        "equivalent_diameter",
        "feret_diameter_max",
        "major_axis_length",
        "minor_axis_length",
    ]:
        df[m] = df[m] * res

    # Average chromatin packing ratio p = DNA mass / Volume
    # For 3D objects, the "area attribute" is actually the volume
    df["volume"] = df["area"] * res**3
    df["bbox_volume"] = df["bbox_area"] * res**3
    df["convex_volume"] = df["convex_area"] * res**3
    df["filled_volume"] = df["filled_area"] * res**3
    # Clean the columns out of the area variables
    df.drop(
        ["area", "bbox_area", "convex_area", "filled_area"],
        inplace=True,
        axis=1,
    )

    # Dimensions in physical units
    df["width"] = (df["bbox-3"] - df["bbox-0"]) * res
    df["length"] = (df["bbox-4"] - df["bbox-1"]) * res
    df["height"] = (df["bbox-5"] - df["bbox-2"]) * res

    # Count mask height in Z for each X slide (YZ), and compute the average
    height_stds_x = [np.count_nonzero(m, axis=1).std() for m in mask]
    df["height_deviation"] = np.array(height_stds_x).mean() * res

    # Elongation is the ratio length/width of the bounding box
    df["elongation"] = df["length"] / df["width"]
    df["aspect_ratio"] = df["height"] / df["length"]

    # Invaginations. Solidity is *ObjectArea/ConvexHullArea*
    df["invagination_prop"] = 1 - df["solidity"]

    # Get surface area using Didac's function
    df["surface_area"] = get_surface_area(mask) * res**2

    # sphericity = (36π(V^2)) ^ (1/3) / A
    # np.cbrt stands for cube root (x^(1/3))
    df["sphericity"] = (
        np.cbrt(36 * np.pi * (df["volume"] ** 2)) / df["surface_area"]
    )

    df["surface/volume"] = df["surface_area"] / df["volume"]

    # ---------gets distance near nuclear border
    dist2bord = distance_transform_edt(mask)
    df["int_0.0-0.5"] = get_intensity_in_distance(
        im, mask, dist2bord, distance_range=(0, 5)
    )
    df["int_1.0-1.5"] = get_intensity_in_distance(
        im, mask, dist2bord, distance_range=(10, 15)
    )
    df["int_ratio"] = df["int_1.0-1.5"] / df["int_0.0-0.5"]

    # Excess of area compared to a sphere
    # df['EOA'] = excess_of_area_ellipse(df)

    return df


def get_DIRs_metrics_2D(DIRs_mask, im, res):
    """Extract features from a 2D nuclear image and mask"""
    df = pd.DataFrame(
        ski.measure.regionprops_table(
            DIRs_mask.astype(np.uint8), im, properties=properties_2d
        )
    )

    # turn pixel units into um units
    for m in [
        "equivalent_diameter",
        "feret_diameter_max",
        "major_axis_length",
        "minor_axis_length",
        "perimeter",
        "perimeter_crofton",
    ]:
        df[m] = df[m] * res

    for m in ["area", "bbox_area", "convex_area", "filled_area"]:
        df[m] = df[m] * (res**2)

    # Dimensions in physical units
    df["width"] = (df["bbox-2"] - df["bbox-0"]) * res
    df["height"] = (df["bbox-3"] - df["bbox-1"]) * res

    df["area/perimeter"] = df["area"] / df["perimeter"]

    # Roundness or circularity - avoid values higher than 1 (infinity)
    roundness = (4.0 * np.pi * df["area"]) / (df["perimeter"] ** 2)
    df["roundness"] = min(1.0, roundness.values[0])
    df["compactness"] = (4.0 * np.pi * df["area"]) / (df["perimeter"] ** 2)

    # Elongation is the ratio length/width of the bounding box
    df["elongation"] = df["height"] / df["width"]

    # Excess of perimeter compared to an ellipse
    df["EOP"] = excess_of_perimeter_ellipse(df)

    dims = range(len(DIRs_mask.shape))

    # assigns condensate domain label
    df["domain_num"] = 0
    for b in df.index:
        centroid_coords = [
            np.round(df.loc[b, f"centroid-{d}"]).astype(np.uint8) for d in dims
        ]
        df.loc[b, "domain_num"] = DIRs_mask[tuple(centroid_coords)]

    # ---------only keeps largest cluster in domain
    df["include"] = True
    for domain in np.unique(df.domain_num):
        (i,) = np.where(df.domain_num == domain)
        if len(i) > 1:
            ii_out = np.where(df.area[i] < max(df.area[i]))
            df.loc[i[ii_out], "include"] = False

    return df


def get_DIRs_metrics_3D(DIRs_mask, im, res):
    """Extract features from a 3D nuclear image and DIRs mask"""

    df = pd.DataFrame(
        ski.measure.regionprops_table(
            DIRs_mask.astype(np.uint8), im, properties=properties_3d_DIR
        )
    )

    # turn pixel units into um units
    for m in ["equivalent_diameter", "major_axis_length"]:
        df[m] = df[m] * res

    # Average chromatin packing ratio p = DNA mass / Volume
    # For 3D objects, the "area attribute" is actually the volume
    df["volume"] = df["area"] * res**3
    df["bbox_volume"] = df["bbox_area"] * res**3
    df["convex_volume"] = df["convex_area"] * res**3
    df["filled_volume"] = df["filled_area"] * res**3
    # Clean the columns out of the area variables
    df.drop(
        ["area", "bbox_area", "convex_area", "filled_area"],
        inplace=True,
        axis=1,
    )

    # Dimensions in physical units
    df["width"] = (df["bbox-3"] - df["bbox-0"]) * res
    df["length"] = (df["bbox-4"] - df["bbox-1"]) * res
    df["height"] = (df["bbox-5"] - df["bbox-2"]) * res

    # Elongation is the ratio length/width of the bounding box
    df["elongation"] = df["length"] / df["width"]
    df["aspect_ratio"] = df["height"] / df["length"]

    # Get surface area using Didac's function
    df["surface_area"] = get_surface_area(DIRs_mask) * res**2

    # phericity = (36π(V^2)) ^ (1/3) / A
    # np.cbrt stands for cube root (x^(1/3))
    df["sphericity"] = (
        np.cbrt(36 * np.pi * (df["volume"] ** 2)) / df["surface_area"]
    )

    df["surface/volume"] = df["surface_area"] / df["volume"]

    dims = range(len(DIRs_mask.shape))

    # assigns condensate domain label
    df["domain_num"] = 0
    for b in df.index:
        centroid_coords = [
            np.round(df.loc[b, f"centroid-{d}"]).astype(np.uint8) for d in dims
        ]
        df.loc[b, "domain_num"] = DIRs_mask[tuple(centroid_coords)]

    # ---------only keeps largest cluster in domain
    df["include"] = True
    for domain in np.unique(df.domain_num):
        (i,) = np.where(df.domain_num == domain)
        if len(i) > 1:
            ii_out = np.where(df.volume[i] < max(df.volume[i]))
            df.loc[i[ii_out], "include"] = False

    return df


def filter_DIRs(DIRs_mask, df, query):
    """Given a DIRs mask and DIRs dataframe, filter those depending on query"""
    DIRs_mask_filt = DIRs_mask.copy()

    # Remove DIRs from the mask based on criteria
    df_out = df.query(query)
    for b in df_out.index:
        DIRs_mask_filt[DIRs_mask == b + 1] = 0
        df.drop(index=b, inplace=True)

    df["n"] = df.shape[0]
    df.reset_index(drop=True, inplace=True)

    # Re-label the segmented DIRs in the mask after cleaning
    for new_label, old_label in enumerate(df["label"], 1):
        DIRs_mask_filt[DIRs_mask_filt == old_label] = new_label

    return DIRs_mask_filt, df


def groupby_and_count(df, group1, group2):
    """Perform group by and count for two groups in a pandas data frame"""
    grouped = df.groupby([group1, group2]).size()
    g1_list = [g1 for g1, _ in grouped.index]
    g2_list = [g2 for _, g2 in grouped.index]
    n_list = [grouped[g1][g2] for g1, g2 in grouped.index]

    toplot = pd.DataFrame(
        list(zip(g1_list, g2_list, n_list)), columns=[group1, group2, "n"]
    )
    toplot = pd.pivot(toplot, index=group1, columns=group2, values="n")

    return toplot


def fix_labels(df):
    """
    Fix the label for each HSC biological condition in the dataframe from the
    folder ID name to new names for plotting
    """

    labels_dict = {
        "young": "Young",
        "aged": "Aged",
        "Aged_treated_RhoAi": "Aged + Ri",
        "myeloid_progenitors": "MP",
        "Young_compressed_8um": "8μm",
        "Young_compressed_5um": "5μm",
        "Young_compressed_3um": "3μm",
    }

    for old, new in labels_dict.items():
        df["condition"] = [c.replace(old, new) for c in df["condition"]]

    return df


def silhouette_plot(X, model, ax):
    """Plot silhuette plots for a K-Means model"""

    y_lower = 10
    y_tick_pos_ = []
    sh_samples = silhouette_samples(X, model.labels_)
    sh_score = silhouette_score(X, model.labels_)

    for idx in range(model.n_clusters):
        values = sh_samples[model.labels_ == idx]
        values.sort()
        size = values.shape[0]
        y_upper = y_lower + size
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, values)
        y_tick_pos_.append(y_lower + 0.5 * size)
        y_lower = y_upper + 10

    ax.axvline(
        x=sh_score, color="red", linestyle="--", label="Avg Silhouette Score"
    )
    ax.set_title("Silhouette Plot for {} clusters".format(model.n_clusters))
    l_xlim = max(-1, min(-0.1, round(min(sh_samples) - 0.1, 1)))
    u_xlim = min(1, round(max(sh_samples) + 0.1, 1))
    ax.set_xlim([l_xlim, u_xlim])
    ax.set_ylim([0, X.shape[0] + (model.n_clusters + 1) * 10])
    ax.set_xlabel("silhouette coefficient values")
    ax.set_ylabel("cluster label")
    ax.set_yticks(y_tick_pos_)
    ax.set_yticklabels(str(idx) for idx in range(model.n_clusters))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.1))
    ax.legend(loc="best")
    return ax


def get_p_values(df, features, by_col, group1, group2):

    p_values = {}

    for c in features:
        if c != by_col:
            g1 = df[df[by_col] == group1][c]
            g2 = df[df[by_col] == group2][c]
            _, p_value = mannwhitneyu(g1, g2)
            p_values[c] = p_value

    # Convert dictionary to pandas DataFrame for easier manipulation
    pv_df = pd.DataFrame.from_dict(
        p_values, orient="index", columns=["p_float"]
    )
    pv_df["p_string"] = [f"{p:.5f}" for p in pv_df["p_float"]]
    pv_df["significance"] = "ns"

    pv_df["<0.05"] = pv_df["p_float"] < 0.05
    pv_df.loc[pv_df["<0.05"] == True, "significance"] = "*"

    pv_df["<0.01"] = pv_df["p_float"] < 0.01
    pv_df.loc[pv_df["<0.01"] == True, "significance"] = "**"

    pv_df["<0.001"] = pv_df["p_float"] < 0.001
    pv_df.loc[pv_df["<0.001"] == True, "significance"] = "***"

    return pv_df


def get_p_values_clusters(df, features, by_col, group):

    p_values = {}

    for c in features:
        if c != by_col:
            g1 = df[df[by_col] == group][c]
            g2 = df[df[by_col] != group][c]
            _, p_value = mannwhitneyu(g1, g2)
            p_values[c] = p_value

    # Convert dictionary to pandas DataFrame for easier manipulation
    pv_df = pd.DataFrame.from_dict(
        p_values, orient="index", columns=["p_float"]
    )
    pv_df["p_string"] = [f"{p:.5f}" for p in pv_df["p_float"]]
    pv_df["significance"] = "ns"

    pv_df["<0.05"] = pv_df["p_float"] < 0.05
    pv_df.loc[pv_df["<0.05"] == True, "significance"] = "*"

    pv_df["<0.01"] = pv_df["p_float"] < 0.01
    pv_df.loc[pv_df["<0.01"] == True, "significance"] = "**"

    pv_df["<0.001"] = pv_df["p_float"] < 0.001
    pv_df.loc[pv_df["<0.001"] == True, "significance"] = "***"

    return pv_df


def calculate_logfoldchange(X):

    X_mean = pd.concat(
        [
            pd.DataFrame(X.mean(), columns=["mean"]),
            X.groupby("Cluster").mean().T,
        ],
        axis=1,
    )

    X_mean.drop(columns=["mean"], inplace=True)
    X_log2fc = X_mean.copy()

    # Make all numbers in range non-negative to work with logarithms
    X_min = np.amin(np.amin(X_mean))
    X_mean -= X_min

    for col in X_mean:
        X_mean_notcol = np.mean(X_mean.loc[:, X_mean.columns != col], axis=1)
        X_log2fc[col] = np.log2(X_mean[col]) - np.log2(X_mean_notcol)

    return X_log2fc
