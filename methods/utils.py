import numpy as np
import tensorflow as tf
import re


def fix_repeated_metric_names(history):
    # Regular expression pattern to match an underscore followed by a number
    pattern = r"_\d+"

    rename_dic = {}

    # Iterate through the metrics and modify the matching elements
    for k in history.keys():
        if re.search(pattern, k):
            # Replace the matched substring with an empty string
            new_k = re.split(pattern, k)[0]
            rename_dic[k] = new_k

    new_hist = {rename_dic.get(k, k): v for k, v in history.items()}

    return new_hist


def get_id_from(files, level="nucleus", extension="png"):
    """
    Extract unique nucleus identifiers from a list of file names.

    This function searches for file names containing a pattern "_nuc" and
    extracts unique nucleus identifiers by removing the "_nuc" suffix.

    Args:
        files (list of str): A list of file names to process.

    Returns:
        list of str: A list of unique nucleus identifiers.
    """
    # Regular expression pattern to match
    if level == "nucleus":
        pattern = r"_\d+\." + extension
    elif level == "batch":
        pattern = r"_nuc"
    else:
        print("level argument must one either nucleus or batch")

    new_names = []

    # For each file name
    for f in files:
        # If it matches the pattern "_nuc"
        if re.search(pattern, f):
            # Split the file name at the "_nuc" pattern and keep the first part
            new_names.append(re.split(pattern, f)[0])

    # Convert the list to a set to remove duplicates, then back to a list
    return new_names


def get_unique_from(files, level="nucleus"):
    """
    Extract unique nucleus identifiers from a list of file names.

    This function searches for file names containing a pattern "_nuc" and
    extracts unique nucleus identifiers by removing the "_nuc" suffix.

    Args:
        files (list of str): A list of file names to process.

    Returns:
        list of str: A list of unique nucleus identifiers.
    """
    # Regular expression pattern to match
    if level == "nucleus":
        pattern = r"_\d+\.png"
    elif level == "batch":
        pattern = r"_nuc"
    else:
        print("level argument must one either nucleus or batch")

    new_names = []

    # For each file name
    for f in files:
        # If it matches the pattern "_nuc"
        if re.search(pattern, f):
            # Split the file name at the "_nuc" pattern and keep the first part
            new_names.append(re.split(pattern, f)[0])

    # Convert the list to a set to remove duplicates, then back to a list
    return list(set(new_names))


def get_max_size(npz_list, n_dim=3):
    """
    Get the maximum size (shape) among a list of Numpy arrays.

    This function iterates through a list of Numpy arrays stored in .npz files,
    extracts their shapes, and returns the maximum size among all arrays.

    Args:
        npz_list (list of str): A list of file paths to .npz files containing Numpy arrays.
        n_dim (int): The number of dimensions expected in the arrays (default is 3).

    Returns:
        tuple of int: A tuple representing the maximum size of the arrays.
    """
    # Initialize an empty array to store sizes
    array_sizes = np.empty(shape=(len(npz_list), n_dim))

    # Iterate through the .npz files
    for i, nuc_path in enumerate(npz_list):
        img = np.load(nuc_path)["img"]
        array_sizes[i] = img.shape

    # Find the maximum size among all arrays
    max_size = array_sizes.max(axis=0)

    # Convert the resulting tuple of floats to integers
    max_size = tuple(int(x) for x in max_size)

    return max_size


def center_and_pad_nuc(img, goal_size):
    """
    Center and pad a Numpy array to achieve a target size.

    This function takes a Numpy array 'img' and adjusts its size to match the
    specified 'goal_size'. It centers the input array and pads it with zeros
    as needed to reach the desired dimensions.

    Args:
        img (numpy.ndarray): The input Numpy array to be centered and padded.
        goal_size (tuple of int): A tuple specifying the target size (shape)
            that the 'img' should be adjusted to.

    Returns:
        numpy.ndarray: The centered and padded Numpy array.
    """
    # Calculate the differences in dimensions
    dim_diffs = np.array(goal_size) - np.array(img.shape)

    # Create a list of tuples specifying the number of zeros to add before
    # and after the nucleus in each dimension, ensuring even distribution
    pad_width = [(d // 2, (d // 2) + (d % 2)) for d in dim_diffs]

    # Pad the 'img' array with zeros based on 'pad_width'
    return np.pad(img, pad_width, mode="constant", constant_values=0)


def get_n_duplicates(lst):
    """Return the number of duplicates found in a list of elements"""
    return len(lst) - len(set(lst))


def get_common(l1, l2):
    """Return the common elements between two lists"""
    return list(set(l1).intersection(l2))


def make_batch(input_list, batch_size):
    grouped_lists = []
    for i in range(0, len(input_list), batch_size):
        group = input_list[i : i + batch_size]
        grouped_lists.append(group)
    return grouped_lists


def dataset_from_partition(
    imgs_dir,
    partition,
    channel_mode,
    label_mode,
    image_size,
    batch_size,
    seed=2023,
):
    ds = tf.keras.utils.image_dataset_from_directory(
        f"{imgs_dir}/{partition}/",
        color_mode=channel_mode,
        labels="inferred",
        label_mode=label_mode,
        interpolation="bilinear",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
    )

    nuc_ids = get_id_from(ds.file_paths, level="nucleus")
    nuc_ids = [s.split("/")[-1] for s in nuc_ids]
    if batch_size > 1:
        nuc_ids = make_batch(nuc_ids, batch_size=batch_size)

    return nuc_ids, ds
