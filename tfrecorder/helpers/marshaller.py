import tensorflow as tf
import numpy as np
import abc
import csv
from collections import OrderedDict







class Example:

    class Field:
        """
        Simple mapping of google.protobuf.type_pb2 to our name space for convenience.
        Commented fields are not supported here (although they are supported by protobuf).
        """

        TYPE_BOOL = 8
        TYPE_ARRAY_FLOAT32 = 12
        TYPE_ARRAY_INT32 = 121
        TYPE_DOUBLE = 2 # NOTE: and not 1 as in the protobuf, because tfrecords only store float32, so we treat double as float
        TYPE_FLOAT = 2
        TYPE_INT32 = 5
        TYPE_INT64 = 3
        TYPE_STRING = 9
    #TYPE_MESSAGE = 11
    #TYPE_SFIXED32 = 15
    #TYPE_SFIXED64 = 16
    #TYPE_SINT32 = 17
    #TYPE_SINT64 = 18
    #TYPE_GROUP = 10
    #TYPE_ENUM = 14
    #TYPE_FIXED32 = 7
    #TYPE_FIXED64 = 6
    #TYPE_UINT32 = 13
    #TYPE_UINT64 = 4

    def __init__(self):

        self.proto_list = []
        self.proto = None


    def load(self, **kwargs):
        """
        To be implemented by concrete subclass.
        Loads the data of the current example into memory.

        Returns:
            -

        Raises:
            IgnoreExampleException: if the data can not be loaded.

        """
        pass #raise NotImplementedError('To be implemented by concrete subclass.')

    def release(self):
        """
        It is quiet easy to get memory leaks when loading the data of examples that are retain in a list.
        To avoid this, we release the tfrecordable data.

        Override for custom implementation.
        """
        for k in self.get_tfrecordable_attribute_names():

            # here k is the getter func name of each attribute marked as @tfrecordable
            setattr(self, k, None)



    def split(self, **kwargs):
        """
        To be implemented by concrete subclass.
        Optionally splits this example data into chunks, and create one Example object per chunk. By default, simply
        returns None.

        Returns:
            examples: list, of Example objects.

        """
        return None


    @abc.abstractmethod
    def to_csv_row(self):
        """
        To be implemented by concrete subclass.
        This is used to write in a csv file a single row that should be enough to recreate this Example afterward.

        Returns:
            -
        """
        raise NotImplementedError('To be implemented by concrete subclass.')

    @classmethod
    @abc.abstractmethod
    def from_csv_row(cls, row, **kwargs):
        """
        To be implemented by concrete subclass.
        This is used to read from a csv file a single row that should be enough to recreate this Example.

        Returns:
            example: an instance of the Example object
        """
        raise NotImplementedError('To be implemented by concrete subclass.')


    def _to_tf_example_proto(self):
        """
        Lazily creates a protobuf message, and load values of the Example instance's attributes that have been marked as
        @tfrecordable.

        Returns:

        """
        if self.proto:
            return self.proto

        feature = []

        for k, t in self.get_tfrecordable_ordered_dict().items():

            # here k is the getter func name of each attribute marked as @tfrecordable
            v = getattr(self, k)

            if v is None:
                continue # simply ignore unset fields

            if t in [Example.Field.TYPE_BOOL,
                     Example.Field.TYPE_INT32,
                     Example.Field.TYPE_INT64]:

                feature.append((k, get_int64_feature(v)))

            elif t in [Example.Field.TYPE_FLOAT,
                       Example.Field.TYPE_DOUBLE]:

                feature.append((k, get_float_feature(v)))

            elif t in [Example.Field.TYPE_STRING,
                       Example.Field.TYPE_ARRAY_FLOAT32,
                       Example.Field.TYPE_ARRAY_INT32]:

                if type(v) is np.ndarray:
                    # make sure that ndarray are converted to int32/float32 and then bytes if they haven't been before
                    if v.dtype != np.float32 and v.dtype != np.int32:
                        raise TypeError('Only int32 and float32 numpy arrays are supported. Found %s for field %s.' % (v.dtype, k))
                    v = v.tobytes()

                elif type(v) is str:

                    v = v.encode('utf-8')

                else:
                    raise TypeError('Type %s is not supported for bytes message.' % type(v))

                feature.append((k, get_bytes_feature(v)))

            else:
                raise TypeError('Type %s is not supported.' % t)

        # transform into ordered dic
        feature = OrderedDict(feature)
        proto = tf.train.Example(features=tf.train.Features(feature=feature))

        if not proto.IsInitialized():
            raise ValueError('Some attributes of the proto have not been set. Please check: %s.' % proto.UnknownFields())

        self.proto = proto

        return proto

    @classmethod
    def get_tfrecordable_attribute_names(cls):
        return [k for k in cls.get_tfrecordable_ordered_dict().keys()]

    @classmethod
    def get_tfrecordable_ordered_dict(cls):
        return OrderedDict(getattr(cls, 'proto_list'))

    def get_byte_size(self):
        proto = self._to_tf_example_proto()
        return proto.ByteSize()


    def serialize_to_string(self):
        proto = self._to_tf_example_proto()
        return proto.SerializeToString()


    @classmethod
    def parse_from_string(cls, serialized_examples):
        """
        Loads the data stored in tfrecords.
        This method is intended to be called from a TFRecords dataset.

        Args:
            serialized_examples:

        Returns:

        """
        feature_description = {}

        for k, t in cls.get_tfrecordable_ordered_dict().items():

            if t in [Example.Field.TYPE_BOOL,
                     Example.Field.TYPE_INT32,
                     Example.Field.TYPE_INT64]:
                v = tf.io.FixedLenFeature([], tf.int64, default_value=0)

            elif t in [Example.Field.TYPE_FLOAT,
                       Example.Field.TYPE_DOUBLE]:

                v = tf.io.FixedLenFeature([], tf.float32, default_value=0.0)

            elif t in [Example.Field.TYPE_STRING,
                       Example.Field.TYPE_ARRAY_FLOAT32,
                       Example.Field.TYPE_ARRAY_INT32]:
                v = tf.io.FixedLenFeature([], tf.string, default_value='')

            else:
                raise TypeError('Type %s is not supported.' % t)

            feature_description[k] = v

        # here we deserialize the next example of the dataset
        parsed_features = tf.io.parse_single_example(serialized_examples, feature_description)

        # we cast
        result = [] # we iterated the ordered dict to make sure we know the order returned to the tf.Dataset later
        for k, t in cls.get_tfrecordable_ordered_dict().items():

            if t == Example.Field.TYPE_BOOL:
                v = tf.cast(parsed_features[k], tf.bool)

            elif t == Example.Field.TYPE_INT32:
                v = tf.cast(parsed_features[k], tf.int32)

            elif t == Example.Field.TYPE_INT64:
                v = parsed_features[k] # we have stored as int64 already

            elif t == Example.Field.TYPE_FLOAT:
                v = tf.cast(parsed_features[k], tf.float32)

            elif t == Example.Field.TYPE_DOUBLE:
                v = tf.cast(parsed_features[k], tf.float32) # we have stored as float32 anyways

            elif t == Example.Field.TYPE_STRING:
                v = parsed_features[k] # we have stored as b'string already

            elif t == Example.Field.TYPE_ARRAY_INT32:
                v = tf.io.decode_raw(parsed_features[k], tf.int32)

            elif t == Example.Field.TYPE_ARRAY_FLOAT32:
                v = tf.io.decode_raw(parsed_features[k], tf.float32)

            else:
                raise TypeError('Type %s is not supported.' % t)

            result.append(v)

        return tuple(result)




    @staticmethod
    def to_csv_file(filepath, examples):

        with open(filepath, 'a', newline='') as f:
            writer = csv.writer(f)
            for ex in examples:
                writer.writerow(ex.to_csv_row())


    @classmethod
    def from_csv_file(cls, filepath, **kwargs):
        """
        Convenience function to read the content of a csv files describing examples, instantiate them and initialize
        their attributes values.

        Args:
            filepath: str, the csv file.

        Returns:
            examples: list, of Example objects.
        """

        examples = []
        with open(filepath, "r") as f:
            reader = csv.reader(f)
            # instantiate the entire file
            for row in reader:
                examples.append(cls.from_csv_row(row, **kwargs))

        return examples



class IgnoreExampleException(Exception):
    """ Throw this exception when an Example can not be loaded for instance."""
    pass


# utils
def get_bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def get_float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def get_int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


# from google.protobuf.proto_builder import MakeSimpleProtoClass
# from google.protobuf.message_factory import MessageFactory
# mf = MessageFactory()
# pc = MakeSimpleProtoClass(tfrecordable.proto_dict)
# proto = mf.GetPrototype(pc.DESCRIPTOR)
# message = proto()
# message.SerializeToString()




