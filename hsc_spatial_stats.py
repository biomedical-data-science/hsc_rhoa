import os
import numpy as np
import pandas as pd
import skimage as ski
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
from scipy.ndimage import distance_transform_edt, binary_fill_holes


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

interesting_2d = [
    "area",
    "bbox_area",
    "convex_area",
    "major_axis_length",
    "minor_axis_length",
    "mean_intensity",
    "min_intensity",
    "max_intensity",
    "perimeter",
    "height_deviation",
    "solidity",
    "roundness",
    "EOP",
    "area/perimeter",
    "DIRs_area",
    "DIRs_bbox_area",
    "DIRs_convex_area",
    "DIRs_major_axis_length",
    "DIRs_minor_axis_length",
    "DIRs_mean_intensity",
    "DIRs_min_intensity",
    "DIRs_max_intensity",
    "DIRs_orientation",
    "DIRs_perimeter",
    "DIRs_min_distance2border",
    "DIRs_centroid_distance2border",
    "DIRs_mean_distance2border",
    "DIRs_n",
]

properties_3d = [
    "area",
    "bbox_area",
    "convex_area",
    "filled_area",
    "bbox",
    "centroid",
    "major_axis_length",
    "mean_intensity",
    "min_intensity",
    "max_intensity",
    "label",
    "solidity",
    "minor_axis_length",
]

properties_3d_DIR = [
    "area",
    "bbox_area",
    "convex_area",
    "filled_area",
    "bbox",
    "centroid",
    "major_axis_length",
    "mean_intensity",
    "min_intensity",
    "max_intensity",
    "label",
    "solidity",
]

interesting_3d = [
    "volume",
    "extent",
    "equivalent_diameter",
    "bbox_volume",
    "convex_volume",
    "width",
    "length",
    "height",
    "height_deviation",
    "aspect_ratio",
    "major_axis_length",
    "minor_axis_length",
    "mean_intensity",
    "min_intensity",
    "max_intensity",
    "solidity",
    "surface_area",
    "sphericity",
    "DIRs_volume",
    "DIRs_bbox_volume",
    "DIRs_convex_volume",
    "DIRs_width",
    "DIRs_length",
    "DIRs_height",
    "DIRs_aspect_ratio",
    "DIRs_surface_area",
    "DIRs_domain_num",
    "DIRs_major_axis_length",
    "DIRs_sphericity",
    "DIRs_surface/volume",
    "DIRs_mean_intensity",
    "DIRs_min_intensity",
    "DIRs_max_intensity",
    "DIRs_solidity",
    "DIRs_centroid_distance2border",
    "DIRs_mean_distance2border",
    "DIRs_min_distance2border",
    "DIRs_n",
]


def load_czi_image(czi_filepath, channel_name="Ch1-T2"):
    """Load a .czi image and return the image data and physical pixel sizes"""
    czi_img = AICSImage(czi_filepath)

    if "get_channel_names" in dir(czi_img):
        channel_names_list = czi_img.get_channel_names()

    elif "channel_names":
        channel_names_list = czi_img.channel_names

    else:
        print("WARNING: Skiping image. No AICSimage cmpatible channel detected")
        return "", ""

    if not any(np.asarray(channel_names_list) == channel_name):
        print("WARNING, Skiping image. No Ch1-T2 Channel.")
        return "", ""

    channel_num = int(np.where(np.asarray(channel_names_list) == channel_name)[0])
    img = czi_img.get_image_data("ZXY", C=channel_num)

    return (img, czi_img.physical_pixel_sizes)


def resize_image(img, pixel_sizes, new_res=0.1, resize_z=True):
    """Given an image and the original pixel sizes, return the image resized
    to the new resolution"""
    new_shape = (
        np.round(img.shape[0] * pixel_sizes.Z / new_res),
        np.round(img.shape[1] * pixel_sizes.X / new_res),
        np.round(img.shape[2] * pixel_sizes.Y / new_res),
    )

    # Check that the number of pixels won't be bigger in X and Y
    for i, _ in enumerate(new_shape[1:], 1):
        if new_shape[i] > img.shape[i]:
            print("Skipping due to outlier X and Y resolution")
            return ""

    # About anti-aliasing:
    # Whether to apply a Gaussian filter to smooth the image prior to
    # downsampling. It is crucial to filter when downsampling the image to
    # avoid aliasing artifacts. If not specified, it is set to True when
    # downsampling an image whose data type is not bool.
    if resize_z:
        img = ski.transform.resize(
            img, new_shape, preserve_range=True, anti_aliasing=True
        )
    else:
        img = ski.transform.resize(
            img,
            (img.shape[0], new_shape[1], new_shape[2]),
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
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros"""
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(mask))
    return img[slices], mask[slices]


def get_nucleus_mask(img, res):
    """Batch calling different segmentation steps
    INPUT:
          img: image numpy 3D array
          res: image resolution
    """

    smooth_img = ski.filters.gaussian(img, sigma=(0.1 / res))
    otsu_thres = ski.filters.threshold_otsu(smooth_img)

    # applies histerisis thresholding of TOLERANCE
    tolerance = 0.2
    mask = ski.filters.apply_hysteresis_threshold(
        smooth_img, otsu_thres * (1 - tolerance), otsu_thres * (1 + tolerance)
    )

    # fills holes
    mask = ski.morphology.binary_closing(mask, ski.morphology.ball(5))
    mask = binary_fill_holes(mask, ski.morphology.ball(5))

    return mask


def plot_seg(img, mask, plane, output_path, contour=True, plot=False):
    """Plots input image with contoured segmentation in a matrix of subplots
    with different slices in the indicated plane
    INPUT:
          img: image numpy 3D array
          mask: segmented image as a binary 3D mask
          plane: plane to plot/slice (e.g. 'XY', )
          output_path: filename for the resulting image
          contour: whether to plot the contour of the nucleus mask
          plot: whether to plot the image also on the terminal
    """

    n_bs = 3
    n_total = 15
    n_cols = int(np.ceil(n_total / n_bs))
    plt.figure(figsize=(15, 10))

    if plane == "XY":
        for i in range(0, n_total):
            plt.subplot(n_bs, n_cols, i + 1)
            plt.imshow(
                img[i * int(np.floor(img.shape[0] / n_total)), :, :],
                vmax=np.max(img.flatten()) * 0.7,
            )

            if contour:
                plt.contour(mask[i * int(np.floor(img.shape[0] / n_total)), :, :])

            plt.title("PLANE : XY")

    elif plane == "ZX":
        for i in range(0, n_total):
            plt.subplot(n_bs, n_cols, i + 1)
            plt.imshow(
                img[:, :, i * int(np.floor(img.shape[2] / n_total))],
                vmax=np.max(img.flatten()) * 0.7,
            )

            if contour:
                plt.contour(mask[:, :, i * int(np.floor(img.shape[2] / n_total))])

            plt.title("PLANE : ZX")

    elif plane == "ZY":
        for i in range(0, n_total):
            plt.subplot(n_bs, n_cols, i + 1)
            plt.imshow(
                img[:, i * int(np.floor(img.shape[1] / n_total)), :],
                vmax=np.max(img.flatten()) * 0.7,
            )

            if contour:
                plt.contour(mask[:, i * int(np.floor(img.shape[1] / n_total)), :])

            plt.title("PLANE : ZY")

    plt.savefig(output_path)
    # Prevent Matlotlib from showing the plot and saturating the log
    plt.close()

    if plot:
        plt.show()


def get_output_filename(input_path, output_path, suffix="XY", extension=".png"):
    """Generates output filename for .png inspection images
    INPUT:
          input_path: input filename of .czi image
          output_path: working output folder
          suffix: suffix to be added before file extension
    """
    filename = (
        os.path.splitext(os.path.basename(input_path))[0] + "_" + suffix + extension
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
    czi_filepath,
    output_filepath,
    channel_name="Ch1-T2",
    new_res=0.1,
    resize=True,
    resize_z=True,
    normalization="z_score",
):
    """batch method concatenating a number of functions to load image,
    re-sample, segment, export, and save .png and numpy images on disk
    INPUT:
          czi_filepath: filename with .czi image
          output_filepath: where to save the resulting image and mask
          channel_name : channel in the .czi file to be used
          new_res: new image resolution for all dimensions through interpolation
          resize: whether to resize the image
          normalization: whether to apply a certain normalization method
    """

    czi_img, pixel_sizes = load_czi_image(czi_filepath, channel_name)

    if not isinstance(czi_img, np.ndarray):
        return "", ""

    # Denoise by using the total variation in the 3D image
    img = ski.restoration.denoise_tv_chambolle(czi_img, weight=0.01)

    # resize image with isotropic interpolation
    if resize:
        img = resize_image(img, pixel_sizes, new_res=new_res, resize_z=resize_z)
        if img == "":
            return "", ""

    # get segmentation mask of the nucleus
    nuc_mask = get_nucleus_mask(img, res=new_res)
    if len(np.unique(nuc_mask)) == 1:
        print("No mask found - skipping")
        return "", ""

    # Remove dimensions that are zeros in the mask
    img, nuc_mask = trim_zeros(img, nuc_mask)
    # Pad in each dimension until getting desired shape (centering the object)
    img, nuc_mask = pad_center([img, nuc_mask])
    img = img * nuc_mask

    # TV-chambolle and resizing make the intensity values very small.
    # Re-scale into the 0-1 interval
    min = np.min(img[nuc_mask.astype(bool)])
    max = np.max(img[nuc_mask.astype(bool)])
    img = (img - min) / (max - min)

    # Normalize using z scores
    if normalization == "z_score":
        img = zscore_normalization(img, mask=nuc_mask)
    elif normalization == "mean":
        img = mean_normalization(img, mask=nuc_mask)
    elif normalization == "none":
        img = img

    print(img.shape)

    # plots
    out_path_xy = get_output_filename_with_subfolder(
        czi_filepath, output_filepath, suffix="XY"
    )
    plot_seg(img, nuc_mask, "XY", contour=True, output_path=out_path_xy)

    out_path_zy = get_output_filename_with_subfolder(
        czi_filepath, output_filepath, suffix="ZY"
    )
    plot_seg(img, nuc_mask, "ZY", contour=True, output_path=out_path_zy)

    # saves segmentation
    np.savez_compressed(
        get_output_filename_with_subfolder(
            czi_filepath, output_filepath, suffix="img", extension=""
        ),
        dapi=img,
        seg=nuc_mask.astype(float),
    )

    return img, nuc_mask


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


def plot_intensity_profile_3D(im, mask, dist2bord, res, max_dist=20, thickness=1.5):
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

    mean_int, nvox, q1_int, q3_int = [np.ones(max_dist) * np.NaN for i in range(4)]

    for lag, dist in enumerate(range(max_dist)):
        coords = tuple(
            np.where(
                (dist2bord < (dist + thickness)) & (dist2bord >= dist) & (mask > 0.1)
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
        np.where((dist2bord < max_dist) & (dist2bord >= min_dist) & (mask > 0.1))
    )

    # If that distance range is not found on the image, NaN is returned
    return np.mean(im[coords])


def get_intensity_by_distance(im, mask, dist2bord, res, max_dist=20, thickness=1.5):
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

        df.loc[b, "centroid_distance2border"] = dist2bord[tuple(centroid_coords)]

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
    df.drop(["area", "bbox_area", "convex_area", "filled_area"], inplace=True, axis=1)

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
    df["sphericity"] = np.cbrt(36 * np.pi * (df["volume"] ** 2)) / df["surface_area"]

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
    df.drop(["area", "bbox_area", "convex_area", "filled_area"], inplace=True, axis=1)

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
    df["sphericity"] = np.cbrt(36 * np.pi * (df["volume"] ** 2)) / df["surface_area"]

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
        "old": "Aged",
        "treated_RhoAi": "Aged + Ri",
        "compressed_8um": "8μm",
        "compressed_5um": "5μm",
        "compressed_3um": "3μm",
    }

    for old, new in labels_dict.items():
        df["condition"] = [c.replace(old, new) for c in df["condition"]]

    return df
