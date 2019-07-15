import csv
import os
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer

tf.compat.v1.enable_eager_execution()


def preprocess_image(image):
    """
    Decode a jpeg and normalize it to [0, 1] range.

    Parameters
    ----------
    image : tf tensor
        Raw image as a tf tensor

    Returns
    -------
        tf tensor in float32
    """
    image = tf.io.decode_jpeg(image, dct_method="INTEGER_ACCURATE")
    image = tf.cast(image, tf.float32)
    image /= 255.0  # normalize to [0,1] range
    return image


def load_and_preprocess_image(path):
    """
    Read a jpeg and normalize it to [0, 1] range.

    Parameters
    ----------
    path : str
        Path to the jpeg to read

    Returns
    -------
        tf tensor in float32
    """
    image = tf.io.read_file(path)
    return preprocess_image(image)


def get_dataset(image_dir, labels_csv):
    """
    Prepare the dataset for training. Images are normalized to [0, 1] range. Labels are n-hot encoded.

    Parameters
    ----------
    image_dir : str
        Path to the directory containing the images
    labels_csv : str
        Path to the csv file

    Returns
    -------
        tf.data.ZipDataset
            Elements of the dataset are structured (image, labels)
    """
    samples = []
    with open(labels_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip first row as it is column names
        for row in reader:
            samples.append((row[0], os.path.join(image_dir, f'{row[0]}.jpg'), row[1].split(' ')))
            if not os.path.exists(samples[-1][1]):
                print(f"WARNING: {samples[-1][1]} does not exist")

    # Create labels list
    labels = np.unique(np.concatenate(np.array(samples)[:, 2]))

    binarizer = MultiLabelBinarizer(classes=labels)
    binarizer.fit(labels)

    # Transform all labels
    transformed_labels = list(map(lambda x: binarizer.transform([x[2]]).squeeze(), samples))

    # Create tensorflow dataset
    img_ds = tf.data.Dataset.from_tensor_slices([s[1] for s in samples])
    img_ds = img_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(transformed_labels, tf.int64))

    img_label_ds = tf.data.Dataset.zip((img_ds, label_ds))

    return img_label_ds


def get_labels_distribution(labels_csv, combination=False):
    """
    Returns a dict where the keys are the labels and the values are the number of occurences.

    Parameters
    ----------
    labels_csv : str
        Path to the csv file
    combination : bool, default False
        If True, returns the combination of labels as they appear instead of splitting them

    Returns
    -------
        dict
    """
    # Read the csv
    sample_labels = []
    with open(labels_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # Skip first row as it is column names
        for row in reader:
            if combination:
                sample_labels.append(row[1])
            else:
                sample_labels.append(row[1].split(' '))

    # Count labels occurences
    if combination:
        labels = Counter(x for x in sample_labels)
    else:
        labels = Counter(x for xs in sample_labels for x in set(xs))

    return labels


def write_csv(csvpath, header, data):
    """
    Write a csv file according to a specific header and data.

    Parameters
    ----------
    csvpath : str
        Path to the csv file that will ve written
    header : list of str
        The header of the csv file
    data : list of list
        The rows of the csv file
    """
    with open(csvpath, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(header)
        writer.writerows(data)


def split_train_val_test(labels_csv, output_dir, train_size=0.6, val_size=0.2, seed=0):
    """
    Split the data in training/validation/testing sets and write each set in a different csv.

    Parameters
    ----------
    labels_csv : str
        Path to the csv file that contains the data to be split
    output_dir : str
        Path to the directory where the new csv will be written
    train_size : float in ]0 ; 1[, default 0.6
        Proportion of the data that will be in the training set
    val_size : float in ]0 ; 1[, default 0.2
        Proportion of the data that will be in the validation set
    seed: int, default 0
        The seed that will be used to shuffle the data

    Note
    ----
    The output csv will be named as:
    `os.path.join(output_dir, f'{set}_seed_{seed}.csv')`

    The proportion of the data in the test set is defined as:
    `test_size = 1 - (train_size + val_size)`

    Combinations of labels with less than 3 elements will be put in the test set.

    Returns
    -------
        tuple of 3 str
        The tuple contains the path to the csv that were written organised as: (train_csv, val_csv, test_csv)

    Raises
    ------
    ValueError
        If train_size not in ]0 ; 1[
        If val_size not in ]0 ; 1[
        If train-size + val_size >= 1
    """
    if not 0 < train_size < 1:
        raise ValueError('train_size not in ]0 ; 1[')
    if not 0 < val_size < 1:
        raise ValueError('val_size not in ]0 ; 1[')
    if train_size + val_size >= 1:
        raise ValueError('train-size + val_size >= 1')

    labels_dist = get_labels_distribution(labels_csv, combination=True)
    current_labels_dist = labels_dist.copy()

    # Read the data
    samples = []
    with open(labels_csv) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        header = next(reader)
        for row in reader:
            samples.append(row)

    # Shuffle the data
    rs = np.random.RandomState(seed=seed)
    rs.shuffle(samples)

    # Split the data
    train_set = []
    val_set = []
    test_set = []
    for s in samples:
        if labels_dist[s[1]] < 3:
            test_set.append(s)
        else:
            leftover = float(current_labels_dist[s[1]]) / float(labels_dist[s[1]])
            if leftover < train_size:
                train_set.append(s)
            elif leftover < train_size + val_size:
                val_set.append(s)
            else:
                test_set.append(s)
            current_labels_dist[s[1]] -= 1

    # Write the split datasets
    train_csv = os.path.join(output_dir, f'train_seed_{seed}.csv')
    val_csv = os.path.join(output_dir, f'val_seed_{seed}.csv')
    test_csv = os.path.join(output_dir, f'test_seed_{seed}.csv')

    write_csv(train_csv, header, train_set)
    write_csv(val_csv, header, val_set)
    write_csv(test_csv, header, test_set)

    return train_csv, val_csv, test_csv
