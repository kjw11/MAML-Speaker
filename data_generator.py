import tensorflow as tf
import numpy as np
import os
# from matplotlib import pyplot as plt
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


IMAGENET_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)


class ImageDataGenerator(object):

    def __init__(self, txt_file, batch_size, num_classes, shuffle=True, buffer_size=1000):

        """Create a new ImageDataGenerator.
        Receives a path string to a text file, where each line has a path string to an image and
        separated by a space, then with an integer referring to the class number.

        Args:
            txt_file: path to the text file.
            mode: either 'training' or 'validation'. Depending on this value, different parsing functions will be used.
            batch_size: number of images per batch.
            num_classes: number of classes in the dataset.
            shuffle: wether or not to shuffle the data in the dataset and the initial file list.
            buffer_size: number of images used as buffer for TensorFlows shuffling of the dataset.

        Raises:
            ValueError: If an invalid mode is passed.
        """

        self.txt_file = txt_file
        self.num_classes = num_classes

        # retrieve the data from the text file
        self._read_txt_file()

        # number of samples in the dataset
        self.data_size = len(self.labels)

        # initial shuffling of the file and label lists together
        if shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.img_paths, self.labels))

        # convert npy to tensor
        data = data.map(self._parse_function_npy, num_parallel_calls=12)

        data = data.batch(batch_size)
        data = data.prefetch(buffer_size=1)

        self.data = data

    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        self.labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                items = line.split(' ')
                self.img_paths.append(os.path.join(items[0]))
                self.labels.append(int(items[1]))

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        labels = self.labels
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        self.labels = []
        for i in permutation:
            self.img_paths.append(path[i])
            self.labels.append(labels[i])

    def read_npy_file(self, filename):
        data = np.load(filename)
        # data = tf.convert_to_tensor(data)
        return data

    def _parse_function_npy(self, filename, label):
        """Input parser for npy filename list"""
        one_hot = tf.one_hot(label, self.num_classes)

        # load npy file
        data = tf.py_func(self.read_npy_file, [filename], tf.float32)
        data = tf.reshape(data, [8, 8, 8])
        return data, one_hot, label

