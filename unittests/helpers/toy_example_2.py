import numpy as np
import os

from tfrecorder.helpers.decorator import tfrecordable
from tfrecorder.helpers.marshaller import Example


class ToyExample2(Example):
    """
    This class illustrate the case where have an ndarray as source and as target (e.g. transformation task).
    """

    def __init__(self,
                 label,
                 name=None,
                 ):

        super(ToyExample2, self).__init__()

        self._label = label
        self.name = name

        self._src_data = None # we dont load data in memory when instantiating
        self._tgt_data = None # we dont load data in memory when instantiating


    # CAUTION: the attributes must be declared in the same order than expected when parsing in tf.Dataset
    @tfrecordable(dtype=Example.Field.TYPE_STRING)
    def label(self):
        return self._label

    @label.setter
    def label(self, val):
        self._label = val

    @tfrecordable(dtype=Example.Field.TYPE_ARRAY_FLOAT32)
    def src_data(self):
        return self._src_data

    @src_data.setter
    def src_data(self, val):
        self._src_data = val

    @tfrecordable(dtype=Example.Field.TYPE_ARRAY_FLOAT32)
    def tgt_data(self):
        return self._tgt_data

    @tgt_data.setter
    def tgt_data(self, val):
        self._tgt_data = val


    def load(self, src_data_dirpath=None, tgt_data_dirpath=None, **kwargs):
        """
        Loads the data in memory, and possibly pre-process it.
        """
        self.src_data = np.load(os.path.join(src_data_dirpath, '%s.npy' % self.name))
        self.tgt_data = np.load(os.path.join(tgt_data_dirpath, '%s.npy' % self.name))


    def split(self, chunk_size_in_bins, **kwargs):
        """
		Splits this example data into chunks, and create one Example object per chunk. By default, simply
		returns a list containing only this Example object.

		Returns:
			examples: list, of Example objects.

		"""
        src_data = self.src_data
        tgt_data = self.tgt_data
        assert src_data.shape[0] == tgt_data.shape[0] >= chunk_size_in_bins, \
            "Data is too short to be chunked (%d vs. %d)." % (src_data.shape[0], chunk_size_in_bins)

        chunked_examples = []
        j = 0
        for i in range(0, src_data.shape[0], chunk_size_in_bins):
            chunked_example = ToyExample2('%s_%d' % (self._label, j))
            chunked_example.src_data = src_data[i:i+chunk_size_in_bins]
            chunked_example.tgt_data = tgt_data[i:i+chunk_size_in_bins]
            chunked_examples.append(chunked_example)
            j += 1

        return chunked_examples


    def to_csv_row(self):
        """
        This is used to write in a csv file a single row that should be enough to recreate this Example afterward.

        Returns:
            -
        """
        return [self._label, self.name]


    @staticmethod
    def from_csv_row(row, **kwargs):
        """
        This is used to read from a csv file a single row that should be enough to recreate this Example.
        Don't forget to cast, as the csv reader returns strings only.

        Returns:
            example: an instance of the Example object
        """
        return ToyExample2(label=row[0],
                           name=row[1])


    def __repr__(self):
        return 'ToyExample2 {name: %s, label: %s}' % (self.name, self._label)