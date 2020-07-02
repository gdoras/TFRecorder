import numpy as np
import os

from tfrecorder.helpers.decorator import tfrecordable
from tfrecorder.helpers.marshaller import Example


class ToyExample1(Example):
    """
    This class illustrate the case where have an ndarray as source and a label or probability as target
    (e.g. classification task).
    """

    def __init__(self,
                 name,
                 label,
                 likelihood,
                 data_filepath,
                 ):

        super(ToyExample1, self).__init__()
        self._name = name
        self._label = label
        self._likelihood = likelihood
        self._data = None # we dont load data in memory when instantiating

        self.data_filepath = data_filepath

    # CAUTION: the attributes must be declared in the same order than expected when parsing in tf.Dataset
    @tfrecordable(dtype=Example.Field.TYPE_STRING)
    def name(self):
        return self._name

    @name.setter
    def name(self, val):
        self._name = val

    @tfrecordable(dtype=Example.Field.TYPE_INT32)
    def label(self):
        return self._label

    @label.setter
    def label(self, val):
        self._label = val

    @tfrecordable(dtype=Example.Field.TYPE_FLOAT)
    def likelihood(self):
        return self._likelihood

    @likelihood.setter
    def likelihood(self, val):
        self._likelihood = val

    @tfrecordable(dtype=Example.Field.TYPE_ARRAY_FLOAT32)
    def data(self):
        return self._data

    @data.setter
    def data(self, val):
        self._data = val


    def load(self, add_row=False, **kwargs):
        """
        Loads the data in memory, and possibly pre-process it.

        Returns:

        """
        data = np.load(self.data_filepath)
        if add_row:
            pad = np.zeros([1,data.shape[1]], dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)

        self.data = data

    def to_csv_row(self):
        """
        This is used to write in a csv file a single row that should be enough to recreate this Example afterward.

        Returns:
            -
        """
        return [self._name, self._label, self._likelihood]


    @staticmethod
    def from_csv_row(row, data_dirpath=None, **kwargs):
        """
        To be implemented by concrete subclass.
        This is used to read from a csv file a single row that should be enough to recreate this Example.
        Don't forget to cast, as the csv reader returns strings only.

        Returns:
            example: an instance of the Example object
        """
        return ToyExample1(row[0], int(row[1]), float(row[2]), os.path.join(data_dirpath, '%s.npy' % row[0]))


    def __repr__(self):
        return 'Example {name: %s, label: %d, likelihood: %.2f, filepath: %s}' % (
            self._name,
            self._label,
            self._likelihood,
            self.data_filepath)



