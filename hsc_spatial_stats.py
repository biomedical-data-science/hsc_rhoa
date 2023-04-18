import os
import numpy as np
import pandas as pd
import skimage as ski
from matplotlib import pyplot as plt
from aicsimageio import AICSImage
from scipy.ndimage import distance_transform_edt, binary_fill_holes


properties_2d = [
    'area', 'bbox_area', 'convex_area', 'filled_area', 'bbox', 'centroid',
    'weighted_centroid', 'eccentricity', 'equivalent_diameter', 'euler_number', 
    'extent', 'feret_diameter_max', 'inertia_tensor', 'inertia_tensor_eigvals', 
    'major_axis_length', 'minor_axis_length', 'mean_intensity', 'min_intensity',
    'max_intensity', 'label', 'orientation', 'perimeter', 'perimeter_crofton',
    'solidity'
    ]

interesting_2d = [
    'area', 'bbox_area', 'convex_area', 'eccentricity', 'equivalent_diameter',
    'extent', 'feret_diameter_max', 'major_axis_length', 'minor_axis_length',
    'mean_intensity', 'min_intensity', 'max_intensity', 'perimeter',
    'height_deviation', 'solidity', 'chrom_packing_ratio', 'entropy',
    'fractal_dim', 'roundness', 'compactness', 'elongation', 'EOP',
    'invagination_prop', 'area/perimeter', 'intensity_outside_mask',
    'DIRs_area', 'DIRs_bbox_area', 'DIRs_convex_area', 'DIRs_filled_area',
    'DIRs_eccentricity', 'DIRs_equivalent_diameter', 'DIRs_euler_number',
    'DIRs_extent', 'DIRs_feret_diameter_max', 'DIRs_major_axis_length',
    'DIRs_minor_axis_length', 'DIRs_mean_intensity', 'DIRs_min_intensity',
    'DIRs_max_intensity', 'DIRs_orientation', 'DIRs_perimeter',
    'DIRs_perimeter_crofton', 'DIRs_solidity', 'DIRs_min_distance2border',
    'DIRs_centroid_distance2border', 'DIRs_mean_distance2border',  'DIRs_n'
    ]

properties_3d = [
    'area', 'bbox_area', 'convex_area', 'filled_area', 'bbox', 'centroid',
    'weighted_centroid', 'equivalent_diameter', 'extent',
    'inertia_tensor', 'inertia_tensor_eigvals', 'major_axis_length', 
    'mean_intensity', 'min_intensity', 'max_intensity', 
    'label', 'solidity', 'minor_axis_length', 'feret_diameter_max',
    ]

properties_3d_DIR = [
    'area', 'bbox_area', 'convex_area', 'filled_area', 'bbox', 'centroid',
    'weighted_centroid', 'equivalent_diameter', 'extent',
    'inertia_tensor', 'inertia_tensor_eigvals', 'major_axis_length', 
    'mean_intensity', 'min_intensity', 'max_intensity', 
    'label', 'solidity',
    ]

interesting_3d = [
    'volume', 'bbox_volume', 'convex_volume', 'filled_volume', 'extent',
    'equivalent_diameter', 'feret_diameter_max', 'width', 'length', 'height', 
    'height_deviation', 'elongation', 'aspect_ratio', 'major_axis_length', 
    'minor_axis_length', 'mean_intensity', 'min_intensity', 'max_intensity', 
    'solidity', 'chrom_packing_ratio', 'invagination_prop', 'surface_area', 
    'surface/volume', 'sphericity', 'fractal_dim', 'entropy', 'DIRs_volume', 
    'DIRs_bbox_volume', 'DIRs_convex_volume', 'DIRs_filled_volume', 
    'DIRs_width', 'DIRs_length', 'DIRs_height', 'DIRs_elongation', 
    'DIRs_aspect_ratio', 'DIRs_surface_area', 'DIRs_domain_num', 
    'DIRs_major_axis_length',
    'DIRs_sphericity', 'DIRs_surface/volume', 'DIRs_mean_intensity', 
    'DIRs_min_intensity', 'DIRs_max_intensity', 'DIRs_solidity', 
    'DIRs_centroid_distance2border', 'DIRs_mean_distance2border', 
    'DIRs_min_distance2border', 'DIRs_n',
    ]


def load_czi_image(czi_filepath, channel_name='Ch1-T2'):

    czi_img = AICSImage(czi_filepath)

    if "get_channel_names" in dir(czi_img):
        channel_names_list = czi_img.get_channel_names()

    elif "channel_names":
        channel_names_list = czi_img.channel_names

    else: 
        print("WARNING: Skiping image. No AICSimage cmpatible channel detected")
        return '', ''

    if not any(np.asarray(channel_names_list) == channel_name):
        print('WARNING, Skiping image. No Ch1-T2 Channel.')
        return '', ''

    channel_num = int(
        np.where(np.asarray(channel_names_list) == channel_name)[0])
        
    return (czi_img.get_image_data("ZXY", C=channel_num),
            czi_img.physical_pixel_sizes)


def resize_image(img, pixel_sizes, new_res=0.1, resize_z=True):

    new_shape = (
            np.round(img.shape[0]*pixel_sizes.Z/new_res),
            np.round(img.shape[1]*pixel_sizes.X/new_res),
            np.round(img.shape[2]*pixel_sizes.Y/new_res)
    )

    # Check that the number of pixels won't be bigger in X and Y
    for i, _ in enumerate(new_shape[1:], 1):
        if new_shape[i] > img.shape[i]:
            print('Skipping due to outlier X and Y resolution')
            return ''

    # About anti-aliasing:
    # Whether to apply a Gaussian filter to smooth the image prior to 
    # downsampling. It is crucial to filter when downsampling the image to 
    # avoid aliasing artifacts. If not specified, it is set to True when 
    # downsampling an image whose data type is not bool.

    # Antialiasing_sigma:
    # Standard deviation for Gaussian filtering used when anti-aliasing. 
    # By default, this value is chosen as (s - 1) / 2 where s is the 
    # downsampling factor, where s > 1. For the up-size case, s < 1, no 
    # anti-aliasing is performed prior to rescaling.
    if resize_z:
        img = ski.transform.resize(img, new_shape, preserve_range=True, 
                                   anti_aliasing=True)
    else:
        img = ski.transform.resize(img, (img.shape[0], new_shape[1], new_shape[2]),
                                   preserve_range=True, anti_aliasing=True)

    print("Rescaling image. New Image Size: ", img.shape)

    return img


def zscore_normalization(im, mask):

    roi = im[mask.astype(bool)]

    im -= roi.mean()
    im /= roi.std()
    return im


def mean_normalization(im, mask):

    roi = im[mask.astype(bool)]
    im -= roi.mean()
    return im


def pad_center(array_list):

    pad_width = [(3, 3) for n in range(len(array_list[0].shape))]
    return [np.pad(a, pad_width=pad_width) for a in array_list]


def from_zxy_to_xyz(array_list):
    return [np.transpose(a, axes=[1, 2, 0]) for a in array_list]


def trim_zeros(img, mask):
    """Returns a trimmed view of an n-D array excluding any outer
    regions which contain only zeros.
    """
    slices = tuple(slice(idx.min(), idx.max() + 1) for idx in np.nonzero(mask))
    return img[slices], mask[slices]


def get_nucleus_mask(img, res):
    # batch calling different segmentation steps
    # INPUT:
    #       np_image : image numpy 3D array
    # OUTPUT:
    #       mask : segmented image as a binary 3D mask

    smooth_img = ski.filters.gaussian(img, sigma=(0.1/res))
    otsu_thres = ski.filters.threshold_otsu(smooth_img)

    # applies histerisis thresholding of TOLERANCE
    tolerance = 0.2
    mask = ski.filters.apply_hysteresis_threshold(
        smooth_img, otsu_thres*(1 - tolerance), otsu_thres*(1 + tolerance)
    )

    # fills holes
    mask = ski.morphology.binary_closing(mask, ski.morphology.ball(5))
    mask = binary_fill_holes(mask, ski.morphology.ball(5))

    return mask


def plot_seg(img, mask, plane, output_path, contour=True, plot=False):
    # Plots input image with contoured segmentation in a matrix of subplots
    # with different slices in the indicated plane
    # INPUT:
    #       imgage : image numpy 3D array
    #       seg_image : segmented image as a binary 3D mask
    #       plane: plane to plot/slice (e.g. 'XY', )

    n_bs = 3
    n_total = 15
    n_cols = int(np.ceil(n_total/n_bs))
    plt.figure(figsize=(15, 10))

    if plane == 'XY':
        for i in range(0, n_total):

            plt.subplot(n_bs, n_cols, i+1)
            plt.imshow(img[i*int(np.floor(img.shape[0]/n_total)),
                       :, :], vmax=np.max(img.flatten())*0.7)

            if contour:
                plt.contour(
                    mask[i*int(np.floor(img.shape[0]/n_total)), :, :])

            plt.title("PLANE : XY")

    elif plane == 'ZX':
        for i in range(0, n_total):

            plt.subplot(n_bs, n_cols, i+1)
            plt.imshow(img[:, :, i*int(np.floor(img.shape[2]/n_total))],
                       vmax=np.max(img.flatten())*0.7)

            if contour:
                plt.contour(
                    mask[:, :, i*int(np.floor(img.shape[2]/n_total))])

            plt.title("PLANE : ZX")

    elif plane == 'ZY':
        for i in range(0, n_total):

            plt.subplot(n_bs, n_cols, i+1)
            plt.imshow(img[:, i*int(np.floor(img.shape[1]/n_total)),
                       :], vmax=np.max(img.flatten())*0.7)

            if contour:
                plt.contour(
                    mask[:, i*int(np.floor(img.shape[1]/n_total)), :])

            plt.title("PLANE : ZY")

    plt.savefig(output_path)
    # Prevent Matlotlib from showing the plot and saturating the log
    plt.close()

    if plot:
        plt.show()


def get_output_filename(input_path, output_path, suffix='XY', extension='.png'):
    # generates output filename for .png inspection images
    # INPUT:
    #       input_path : input filename of .czi image
    #       output_path : working output folder
    #       suffix: suffix to be added before file extension
    # OUTPUT :
    #       output filepath string

    filename = os.path.splitext(os.path.basename(input_path))[0] +  \
               '_' + suffix + extension

    return os.path.join(output_path, filename.replace(" ", "_"))


def get_output_filename_with_subfolder(input_path, output_path, suffix='XY', 
                                       extension='.png'):
    # generates output filename for .png inspection images adding an 
    # intermediate subfolder
    # INPUT:
    #       input_path : input filename of .czi image
    #       output_path : working output folder
    #       suffix: suffix to be added before file extension
    # OUTPUT :
    #       output filepath string

    filename = os.path.splitext(os.path.basename(input_path))[0] + '_' \
               + os.path.basename(os.path.dirname(input_path)) + '_' \
               + suffix + extension
    return os.path.join(output_path, filename.replace(" ", "_"))
    

def czi_image_preprocessing(czi_filepath, output_filepath,
                            channel_name='Ch1-T2', new_res=0.1, resize=True, 
                            resize_z=True, normalization="z_score"):
    # batch method concatenating a number of functions to load image, 
    # re-sample, segment, export, and save .png and numpy images on disk
    # INPUT:
    #       czi_filepath: filename with .czi image
    #       output_filepath: if not empty (default), then image an numpy 
    #       output is saved
    #       channel_name : channel in the .czi file to be used
    # OUTPUT:
    #       nuc_mask: segmentated nucleus as binary numpy array
    #       img: re-sampled input image as numpy array

    czi_img, pixel_sizes = load_czi_image(czi_filepath, channel_name)

    if not isinstance(czi_img, np.ndarray):
        return '', ''

    # Denoise by using the total variation in the 3D image
    img = ski.restoration.denoise_tv_chambolle(czi_img, weight=0.01)

    # resize image with isotropic interpolation
    if resize:
        img = resize_image(img, pixel_sizes, new_res=new_res,
                           resize_z=resize_z)
        if img == '':
            return '', ''

    # get segmentation mask of the nucleus
    nuc_mask = get_nucleus_mask(img, res=new_res)
    if len(np.unique(nuc_mask)) == 1:
        print('No mask found - skipping')
        return '', ''

    # Remove dimensions that are zeros in the mask
    img, nuc_mask = trim_zeros(img, nuc_mask)
    # Pad in each dimension until getting desired shape (centering the object)
    img, nuc_mask = pad_center([img, nuc_mask])
    img = img*nuc_mask

    # TV-chambolle and resizing make the intensity values very small.
    # Re-scale into the 0-1 interval
    min = np.min(img[nuc_mask.astype(bool)])
    max = np.max(img[nuc_mask.astype(bool)])
    img = (img - min) / (max - min)

    # MAYBE I SHOULD ROUND TO 0 - 255 UINT, TO SAVE MEMORY
    # THE ROI CAN BE EXTRACTED AND INTENSITIES OUTSIDE OF THE MASK ARE USUALLY
    # NAN
    # INTENSITY DISCRETIZATION - i.e. IN 32 BINS?

    # Normalize using z scores
    if normalization == "z_score":
        img = zscore_normalization(img, mask=nuc_mask)
    elif normalization == "mean":
        img = mean_normalization(img, mask=nuc_mask)
    elif normalization == "none":
        img = img
    
    # Move the normalized intensities range back to the original (0, 256)
    # img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 256
    # img = img.astype(np.uint8)
    print(img.shape)

    # plots
    out_path_xy = get_output_filename_with_subfolder(
        czi_filepath, output_filepath, suffix='XY')
    plot_seg(img, nuc_mask, 'XY', contour=True, output_path=out_path_xy)

    out_path_zy = get_output_filename_with_subfolder(
        czi_filepath, output_filepath, suffix='ZY')
    plot_seg(img, nuc_mask, 'ZY', contour=True, output_path=out_path_zy)

    # saves segmentation
    np.savez_compressed(get_output_filename_with_subfolder(
        czi_filepath, output_filepath, suffix='img', extension=""),
        dapi=img, seg=nuc_mask.astype(float)
    )

    return img, nuc_mask


def get_morphometrics_from_largest_DIR(mask, im):
    # gets morphometric b with volume and morphometric values for the larger 
    # DIR in the segmentation
    
    info_table = pd.DataFrame(ski.measure.regionprops_table(
        mask.astype(np.uint8), im,
        properties=[
            'area', 'mean_intensity', 'solidity', 'convex_area', 'centroid',
            'inertia_tensor_eigvals', 'major_axis_length', 'minor_axis_length', 
            'equivalent_diameter',  'weighted_centroid'
            ],
    ))

    b, = np.where(info_table.area == max(info_table.area))
    new_entry = info_table.iloc[b, ]

    nlabels = info_table.shape[0]  # number of labels
    new_entry["nlabels"] = nlabels

    return new_entry


def get_surface_area(mask, sigma=2):
    # reconstructs mesh surface on binary 3d mask and estimates a surface 
    # measure. Marching cubes parameters can be changed. I found these worked 
    # well
    # INPUT
    #       seg_image_morph: segmentated nucleus as binary numpy array
    #       sigma: smoothing of border prior to the marching cubes algorithm 
    #       to avoid overfitting pixelized border
    # OUTPUT
    #       area: measured area of the fitted mesh
    mask_smooth = ski.filters.gaussian(mask, sigma=sigma)
    verts, faces, normals, values = ski.measure.marching_cubes(
        mask_smooth, level=None, step_size=2)

    return ski.measure.mesh_surface_area(verts, faces)


def plot_IProfile_3D(im, mask, dist2bord, file_name=''):
    # Method to plot an Average Mean vs. Distance to edge curve for a single 
    # nucleus with interquartile intervals
    # INPUT:
    #       np_image : image numpy 3D array
    #       seg_image_morph: segmentated nucleus as binary numpy array
    #       dist2bord: numpy array containing all the closest distances to 
    #       the border for each voxel
    #       (euclidian distance transform)
    #       filename: If provided (default is ''), then a .png image is 
    #       exported to the corresponding filename
    
    mean_allint = np.mean(im[mask > 0.1])
    q1_allint = np.quantile(im[mask > 0.1], 0.25)
    q3_allint = np.quantile(im[mask > 0.1], 0.75)

    MAX_DIST = 20
    MIN_DIST = 5
    THICK_INT = 1.5

    mean_int, nvox, q1_int, q3_int = [
        np.ones(MAX_DIST + MIN_DIST) * np.NaN for i in range(4)]

    for lag, dist in enumerate(range(-MIN_DIST, MAX_DIST)):
        if dist < 0:
            i, j, k = np.where((dist2bord < np.abs(dist - THICK_INT))
                               & (dist2bord >= np.abs(dist)) & (mask < 0.2))
        else:
            i, j, k = np.where((dist2bord < (dist + THICK_INT))
                               & (dist2bord >= dist) & (mask > 0.9))
        if i.shape[0] == 0:
            continue

        mean_int[lag] = np.mean(im[i, j, k])/mean_allint
        q1_int[lag] = np.quantile(im[i, j, k], 0.25)/mean_allint
        q3_int[lag] = np.quantile(im[i, j, k], 0.75)/mean_allint
        nvox[lag] = len(i)

    plt.errorbar(x=np.asarray(range(-MIN_DIST, MAX_DIST)), y=mean_int,
                 yerr=[mean_int - q1_int, q3_int - mean_int], fmt='o', 
                 linestyle='-')
    plt.axhline(y=q1_allint/mean_allint, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=q3_allint/mean_allint, color='r', linestyle='--', alpha=0.5)
    plt.title("Mean DAPI Intensity vs. Distance to Border")

    if file_name != '':
        plt.savefig(file_name)
    plt.show()


def plot_intensity_profile_3D(im, mask, dist2bord, res, max_dist=20, 
                              thickness=1.5):

    q1_allint = np.quantile(im[mask > 0.1], 0.25)
    q3_allint = np.quantile(im[mask > 0.1], 0.75)

    mean_int, nvox, q1_int, q3_int = [
        np.ones(max_dist) * np.NaN for i in range(4)]

    for lag, dist in enumerate(range(max_dist)):

        coords = tuple(np.where((dist2bord < (dist + thickness))
                               & (dist2bord >= dist) & (mask > 0.1)))
        if coords[0].shape[0] == 0:
            continue

        mean_int[lag] = np.mean(im[coords])
        q1_int[lag] = np.quantile(im[coords], 0.25)
        q3_int[lag] = np.quantile(im[coords], 0.75)
        nvox[lag] = len(coords[0])

    distances = np.asarray(range(max_dist))*res

    plt.errorbar(x=distances, y=mean_int, fmt='o', linestyle='-',
                 yerr=[mean_int - q1_int, q3_int - mean_int])
    plt.axhline(y=q1_allint, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=q3_allint, color='r', linestyle='--', alpha=0.5)
    plt.title("Mean DAPI Intensity vs. Distance to Border")
    plt.show()


def get_intensity_in_distance(im, mask, dist2bord, distance_range=(0, 10)):

    min_dist, max_dist = distance_range

    coords = tuple(np.where((dist2bord < max_dist)
                               & (dist2bord >= min_dist) & (mask > 0.1)))
                               
    # If that distance range is not found on the image, NaN is returned
    return np.mean(im[coords])


def get_intensity_by_distance(im, mask, dist2bord, res, max_dist=20, 
                              thickness=1.5):

    int_df = pd.DataFrame()

    for dist in range(max_dist):
        
        mean_int = get_intensity_in_distance(
            im, mask, dist2bord, distance_range=(dist, (dist+thickness)))
        df = pd.DataFrame({'distance': dist*res, 'mean_int': [mean_int]})
        # Delete from the dataframe those distances with NaN
        df = df.dropna()
        int_df = pd.concat([int_df, df])

    return int_df


def get_variogram_range(mask, im, res, mode='3D'):

    # estimates variogram in a random subsample of voxels and outputs the 
    # effective range fitted to it
    # INPUT:
    #       np_image : image numpy 3D array
    #       seg_image_morph: segmentated nucleus as binary numpy array
    #       res : image resolution
    # OUTPUT:
    #       range: estimated effective range in micrometers

    coords = np.where(mask > 0.5)

    # random coordinates
    N = 2000
    random_idx = np.random.randint(0, len(coords[0]), N)

    if mode == '3D':
        ii, jj, kk = coords
        values = np.fromiter((im[ii[i], jj[i], kk[i]] for i in random_idx),
                             dtype=float)
    elif mode == '2D':
        ii, jj = coords
        values = np.fromiter((im[ii[i], jj[i]] for i in random_idx),
                             dtype=float)

    coords = np.concatenate(([c[random_idx].reshape(-1, 1) for c in coords]),
                            axis=1)
    coords = coords/(res*100)  # given in micrometers
    #V = skg.Variogram(coords, values, maxlag=40/res*100)

    return V.describe().get('effective_range')


def segment_condensates(im, mask, cluster_thresh=1.8, mode='relative'):

    # Create binary mask with DIRs
    if mode == 'relative':
        nucleus_mean = np.mean(im[mask > 0.5])
        def_thres = nucleus_mean * cluster_thresh
    elif mode == 'absolute':
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

    dims = range(len(DIRs_mask.shape))

    for b in df.index:

        # doesn't compute for non-included
        if not df.include[b]:
            continue

        # computes distance to edge
        centroid_coords = [
            np.round(df.loc[b, f"centroid-{d}"]).astype(np.uint8) 
            for d in dims]

        df.loc[b, "centroid_distance2border"] = dist2bord[
            tuple(centroid_coords)]

        df.loc[b, "mean_distance2border"] = np.mean(
            dist2bord[DIRs_mask == (b + 1)].flatten())
            
        df.loc[b, "min_distance2border"] = np.min(
            dist2bord[DIRs_mask == (b + 1)].flatten())

    return df


def calculate_convex_diff_3D(mask):

    # Plot the difference between the image and the hull
    convex_hull = ski.morphology.convex_hull_image(mask)
    diff = convex_hull - mask
    return np.mean(diff), np.sum(diff), np.sum(diff)/np.sum(convex_hull)


def calculate_convex_diff_2D(mask, plane='XY'):

    diff, total = [], []

    if plane == 'ZX':
        mask = np.transpose(mask, axes=[2, 0, 1])

    if plane == 'ZY':
        mask = np.transpose(mask, axes=[1, 0, 2])

    for mask2d in mask:
        if np.any(mask2d):
            convex_hull = ski.morphology.convex_hull_image(mask2d)
            diff.append(convex_hull - mask2d)
            total.append(convex_hull)

    # CORRECT FOR THE NUMBER OF Z DIMENSIONS
    return np.mean(diff), np.sum(diff), np.sum(diff)/np.sum(total)


# From https://github.com/rougier/numpy-100 (#87)
def boxcount(Z, k):
    S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
            np.arange(0, Z.shape[1], k), axis=1)

    # We count non-empty (0) and non-full boxes (k*k)
    return len(np.where((S > 0) & (S < k*k))[0])


def cubecount(Z, k):
    S = np.add.reduceat(
            np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                                np.arange(0, Z.shape[1], k), axis=1),
                                np.arange(0, Z.shape[2], k), axis=2)

    # We count non-empty (0) and non-full cubes (k*k*k)
    return len(np.where((S > 0) & (S < k*k*k))[0])


# From https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0
# THIS SHOULD BE MENTIONED AND THE AUTHOR
def fractal_dimension(mask, threshold=0.9, mode='2D'):
    """mask should be a two dimensional and binary image"""

    # Minimal dimension of image
    p = min(mask.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    if mode == '2D':
        counts = [boxcount(mask, size) for size in sizes]
    elif mode == '3D':
        counts = [cubecount(mask, size) for size in sizes]

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    
    return -coeffs[0]


def dist(p1, p2):
    return np.sqrt(np.sum(np.square(np.array(p1) - np.array(p2))))


def calculate_ellipse_perimeter(a, b):
    # PI * ( 3*(a + b) - SQRT( (3*a + b) * (a + 3*b) ) )
    return np.pi*(3*(a+b) - np.sqrt((3*a + b)*(a + 3*b)))


def excess_of_perimeter_ellipse(obj):

    ellipse_p = calculate_ellipse_perimeter(obj.major_axis_length/2, 
                                            obj.minor_axis_length/2)
    eop = (obj.perimeter - ellipse_p) / ellipse_p
    return eop


def get_slide(im, mask, mode='middle', plane='XY'):
    """"The incoming array must be 3D in format XYZ"""

    assert len(im.shape) == 3, 'input matrices are not 3D'
    assert im.shape == mask.shape, 'image shape and mask shape are not equal'
    assert plane in ['XY', 'ZX', 'ZY'], 'plane should be XY, ZX or ZY'

    if mode == 'middle':
        # returns the middle slide
        if plane == 'XY':
            mid_zslice = np.floor(im.shape[2]/2)
            im, mask = im[:, :, mid_zslice], mask[:, :, mid_zslice]
        elif plane == 'ZX':
            mid_zslice = np.floor(im.shape[1]/2)
            im, mask = im[:, mid_zslice, :], mask[:, mid_zslice, :] 
        elif plane == 'ZY':
            mid_zslice = np.floor(im.shape[0]/2)
            im, mask = im[mid_zslice, :, :], mask[mid_zslice, :, :]

    elif mode == 'max':
        # return a slide with the maximum value found for each pixel
        if plane == 'XY':
            im, mask = im.max(axis=2), mask.max(axis=2)
        elif plane == 'ZX':  
            im, mask = im.max(axis=1), mask.max(axis=1)
        elif plane == 'ZY':
            im, mask = im.max(axis=0), mask.max(axis=0)

    elif mode == 'largest':
        # returns the slide with the maximum area
        if plane == 'XY':
            max_index = np.argmax(mask.sum(0).sum(0))
            im, mask = im[:, :, max_index], mask[:, :, max_index]
        elif plane == 'ZX':
            max_index = np.argmax(mask.sum(0).sum(1))
            im, mask = im[:, max_index, :], mask[:, max_index, :]
        elif plane == 'ZY':
            max_index = np.argmax(mask.sum(1).sum(1))
            im, mask = im[max_index, :, :], mask[max_index, :, :]
        
    return im, mask


def get_nuclear_metrics_2D(mask, im, res):

    label_mask = ski.measure.label(mask)

    df = pd.DataFrame(ski.measure.regionprops_table(
        label_mask, intensity_image=im, properties=properties_2d
        ))

    df['intensity_outside_mask'] = np.sum(im[~mask]).astype('int64')

    # turn pixel units into um units
    for m in ['equivalent_diameter', 'feret_diameter_max', 'major_axis_length', 
              'minor_axis_length', 'perimeter', 'perimeter_crofton']:
        df[m] = df[m]*res
    
    for m in ['area', 'bbox_area', 'convex_area', 'filled_area']:
        df[m] = df[m]*(res**2)

    # Dimensions in physical units
    df['width'] = (df['bbox-2'] - df['bbox-0']) * res
    df['height'] = (df['bbox-3'] - df['bbox-1']) * res

    df['height_deviation'] = np.count_nonzero(mask, axis=0).std() * res

    df['area/perimeter'] = df['area']/df['perimeter']

    # Average chromatin packing ratio p = DNA mass / Volume
    hsc_dna_mass = 7
    df['chrom_packing_ratio'] = hsc_dna_mass / df['area']

    df['entropy'] = ski.measure.shannon_entropy(im*mask)

    # Roundness or circularity - avoid values higher than 1 (infinity)
    roundness = 4.0 * np.pi * df["area"] / df["perimeter"]**2
    df['roundness'] = min(1., roundness.values[0])

    # RE-WRITE THE FOLLOWING LINES
    # *ObjectArea/Area of circle with the same perimeter*
    df['compactness'] = df["perimeter"]**(2/(4.0 * np.pi * df["area"]))

    # Elongation is the ratio length/width of the bounding box
    df['elongation'] = df['height']/df['width']

    # Invaginations. Solidity is *ObjectArea/ConvexHullArea*
    df['invagination_prop'] = 1 - df['solidity']

    # Fractal dimension
    df['fractal_dim'] = fractal_dimension(mask, mode='2D')

    # Excess of perimeter compared to an ellipse
    df['EOP'] = excess_of_perimeter_ellipse(df)

    # Invaginations. Solidity is *ObjectArea/ConvexHullArea*
    df['invagination_prop'] = 1 - df['solidity']

    return df


def get_nuclear_metrics_3D(mask, im, res):

    label_mask = ski.measure.label(mask)

    df = pd.DataFrame(ski.measure.regionprops_table(
        label_mask, intensity_image=im, properties=properties_3d
        ))

    # turn pixel units into um units
    for m in ['equivalent_diameter', 'feret_diameter_max',
              'major_axis_length', 'minor_axis_length']:
        df[m] = df[m]*res
    
    # Average chromatin packing ratio p = DNA mass / Volume
    # For 3D objects, the "area attribute" is actually the volume
    df['volume'] = df['area'] * res**3
    df['bbox_volume'] = df['bbox_area'] * res**3
    df['convex_volume'] = df['convex_area'] * res**3
    df['filled_volume'] = df['filled_area'] * res**3
    # Clean the columns out of the area variables
    df.drop(['area', 'bbox_area', 'convex_area', 'filled_area'],
            inplace=True, axis=1)

    # Dimensions in physical units
    df['width'] = (df['bbox-3'] - df['bbox-0']) * res
    df['length'] = (df['bbox-4'] - df['bbox-1']) * res
    df['height'] = (df['bbox-5'] - df['bbox-2']) * res

    # Count mask height in Z for each X slide (YZ), and compute the average
    height_stds_x = [np.count_nonzero(m, axis=1).std() for m in mask]
    df['height_deviation'] = np.array(height_stds_x).mean() * res

    # Average chromatin packing ratio p = DNA mass / Volume
    hsc_dna_mass = 7
    df['chrom_packing_ratio'] = hsc_dna_mass / df['volume']

    df['entropy'] = ski.measure.shannon_entropy(im)

    # Elongation is the ratio length/width of the bounding box
    df['elongation'] = df['length']/df['width']
    df['aspect_ratio'] = df['height']/df['length']

    # Invaginations. Solidity is *ObjectArea/ConvexHullArea*
    df['invagination_prop'] = 1 - df['solidity']

    # Fractal dimension
    df['fractal_dim'] = fractal_dimension(mask, mode='3D')

    # Get surface area using Didac's function
    df["surface_area"] = get_surface_area(mask) * res**2

    # sphericity = (36π(V^2)) ^ (1/3) / A
    # np.cbrt stands for cube root (x^(1/3))
    df['sphericity'] = np.cbrt(36*np.pi*(df['volume']**2)) / df['surface_area']

    df['surface/volume'] = df['surface_area']/df['volume']

    # ---------gets distance near nuclear border
    dist2bord = distance_transform_edt(mask)  
    df['int_0.0-0.5'] = get_intensity_in_distance(im, mask, dist2bord, 
                                                  distance_range=(0, 5))
    df['int_1.0-1.5'] = get_intensity_in_distance(im, mask, dist2bord, 
                                                  distance_range=(10, 15))
    df['int_ratio'] = df['int_1.0-1.5']/df['int_0.0-0.5']

    # Excess of area compared to a sphere
    # df['EOA'] = excess_of_area_ellipse(df)

    return df


def get_DIRs_metrics_2D(DIRs_mask, im, res):

    df = pd.DataFrame(ski.measure.regionprops_table(
            DIRs_mask.astype(np.uint8), im, properties=properties_2d))

    # turn pixel units into um units
    for m in ['equivalent_diameter', 'feret_diameter_max', 'major_axis_length', 
              'minor_axis_length', 'perimeter', 'perimeter_crofton']:
        df[m] = df[m]*res
    
    for m in ['area', 'bbox_area', 'convex_area', 'filled_area']:
        df[m] = df[m]*(res**2)

    # Dimensions in physical units
    df['width'] = (df['bbox-2'] - df['bbox-0']) * res
    df['height'] = (df['bbox-3'] - df['bbox-1']) * res

    df['area/perimeter'] = df['area']/df['perimeter']

    df['entropy'] = ski.measure.shannon_entropy(im*DIRs_mask)

    # Roundness or circularity - avoid values higher than 1 (infinity)
    roundness = (4.0 * np.pi * df["area"]) / (df["perimeter"]**2)
    df['roundness'] = min(1., roundness.values[0])

    # RE-WRITE THE FOLLOWING LINES
    # *ObjectArea/Area of circle with the same perimeter*
    # IT IS THE SAME AS ROUNDNESS?
    df['compactness'] = (4.0 * np.pi * df["area"])/(df["perimeter"]**2)

    # Elongation is the ratio length/width of the bounding box
    df['elongation'] = df['height']/df['width']

    # Excess of perimeter compared to an ellipse
    df['EOP'] = excess_of_perimeter_ellipse(df)

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
        i, = np.where(df.domain_num == domain)
        if len(i) > 1:
            ii_out = np.where(df.area[i] < max(df.area[i]))
            df.loc[i[ii_out], "include"] = False

    return df


def get_DIRs_metrics_3D(DIRs_mask, im, res):

    df = pd.DataFrame(ski.measure.regionprops_table(
            DIRs_mask.astype(np.uint8), im, properties=properties_3d_DIR))

    # turn pixel units into um units
    for m in ['equivalent_diameter', 'major_axis_length']:
        df[m] = df[m]*res
    
    # Average chromatin packing ratio p = DNA mass / Volume
    # For 3D objects, the "area attribute" is actually the volume
    df['volume'] = df['area'] * res**3
    df['bbox_volume'] = df['bbox_area'] * res**3
    df['convex_volume'] = df['convex_area'] * res**3
    df['filled_volume'] = df['filled_area'] * res**3
    # Clean the columns out of the area variables
    df.drop(['area', 'bbox_area', 'convex_area', 'filled_area'],
            inplace=True, axis=1)

    # Dimensions in physical units
    df['width'] = (df['bbox-3'] - df['bbox-0']) * res
    df['length'] = (df['bbox-4'] - df['bbox-1']) * res
    df['height'] = (df['bbox-5'] - df['bbox-2']) * res

    # Average chromatin packing ratio p = DNA mass / Volume
    hsc_dna_mass = 7
    df['chrom_packing_ratio'] = hsc_dna_mass / df['volume']

    # Elongation is the ratio length/width of the bounding box
    df['elongation'] = df['length']/df['width']
    df['aspect_ratio'] = df['height']/df['length']

    # Get surface area using Didac's function
    df["surface_area"] = get_surface_area(DIRs_mask) * res**2

    # phericity = (36π(V^2)) ^ (1/3) / A
    # np.cbrt stands for cube root (x^(1/3))
    df['sphericity'] = np.cbrt(36*np.pi*(df['volume']**2)) / df['surface_area']

    df['surface/volume'] = df['surface_area']/df['volume']

    # Invaginations. Solidity is *ObjectArea/ConvexHullArea*
    df['invagination_prop'] = 1 - df['solidity']
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
        i, = np.where(df.domain_num == domain)
        if len(i) > 1:
            ii_out = np.where(df.volume[i] < max(df.volume[i]))
            df.loc[i[ii_out], "include"] = False

    return df


def filter_DIRs(DIRs_mask, df, query):

    DIRs_mask_filt = DIRs_mask.copy()

    # DO BY MASK PROPORTION

    # Remove DIRs from the mask based on criteria
    df_out = df.query(query)
    for b in df_out.index:
        DIRs_mask_filt[DIRs_mask == b + 1] = 0
        df.drop(index=b, inplace=True)

    df['n'] = df.shape[0]

    df.reset_index(drop=True, inplace=True)

    # Re-label the segmented DIRs in the mask after cleaning
    for new_label, old_label in enumerate(df['label'], 1):
        DIRs_mask_filt[DIRs_mask_filt == old_label] = new_label

    return DIRs_mask_filt, df


def groupby_and_count(df, group1, group2):

    grouped = df.groupby([group1, group2]).size()
    g1_list = [g1 for g1, _ in grouped.index]
    g2_list = [g2 for _, g2 in grouped.index]
    n_list = [grouped[g1][g2] for g1, g2 in grouped.index]

    toplot = pd.DataFrame(list(zip(g1_list, g2_list, n_list)),
                          columns=[group1, group2, "n"])
    toplot = pd.pivot(toplot, index=group1, columns=group2, values="n")

    return toplot