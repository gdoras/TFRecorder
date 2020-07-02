import unittest
import math
import numpy as np

from tfrecorder.helpers.decorator import tfrecordable
from tfrecorder.helpers.marshaller import Example

class MarshallerTestCase(unittest.TestCase):
    
    
    def test_example(self):
        """
        Test that a subclass of Example with different types of features produces a protobuf message that can be
        serialized and deserialized as expected.
        """

        feature_bool = True
        feature_int32 = 3
        feature_int64 = 2**63-1
        feature_float = 0.1234
        feature_double = math.pi
        feature_string = 'blabla'
        shape = [3, 4]
        feature_np_array_int32 = np.ones(shape).astype(np.int32)
        feature_np_array_float32 = np.random.random(shape).astype(np.float32)

        toy_example = ToyExample(feature_bool = feature_bool,
                                 feature_int32 = feature_int32,
                                 feature_int64 = feature_int64,
                                 feature_float = feature_float,
                                 feature_double = feature_double,
                                 feature_string = feature_string,
                                 feature_np_array_int32 = feature_np_array_int32,
                                 feature_np_array_float32 = feature_np_array_float32,
                                 )


        # check that the proto names and types have been correctly set
        expected_proto_dict = dict(

            feature_bool = Example.Field.TYPE_BOOL,
            feature_int32 = Example.Field.TYPE_INT32,
            feature_int64 = Example.Field.TYPE_INT64,
            feature_float = Example.Field.TYPE_FLOAT,
            feature_double = Example.Field.TYPE_FLOAT,
            feature_string = Example.Field.TYPE_STRING,
            feature_np_array_int32 = Example.Field.TYPE_ARRAY_INT32,
            feature_np_array_float32 = Example.Field.TYPE_ARRAY_FLOAT32,

        )

        self.assertEqual(ToyExample.get_tfrecordable_ordered_dict(), expected_proto_dict)

        # check that the example has the expected size
        self.assertNotEqual(0, toy_example.get_byte_size())

        # marhall and unmarshall and check we have what was expected
        tensors = ToyExample.parse_from_string(toy_example.serialize_to_string())

        for i, k in enumerate(expected_proto_dict.keys()):

            ev = getattr(toy_example, k)
            t = tensors[i]
            v = t.numpy() # we got tensors so we convert to numpy

            if type(ev) is np.ndarray:
                # the message arrays are stored as bytes, so we have to decode and reshape them first
                if ev.dtype == np.float32 or ev.dtype == np.float64:
                    v = np.frombuffer(v, dtype=np.float32).reshape(shape)
                elif ev.dtype == np.int32 or ev.dtype == np.int64:
                    v = np.frombuffer(v, dtype=np.int32).reshape(shape)
                else:
                    raise ValueError('Type %s is not supported for a numpy array.' % ev.dtype)

                if ev.dtype == np.float64:
                    # we have automatically converted to float32 before serializing
                    # this can not be recovered, so we cast the original value to np.float32
                    ev = ev.astype(np.float32)
                elif ev.dtype == np.int64:
                    ev = ev.astype(np.int32)

                np.testing.assert_array_equal(ev, v)


            elif type(ev) is str:
                # the bytes is string
                v = v.decode("utf-8")
                self.assertEqual(ev, v)

            else:
                if v.dtype == np.float32:
                    ev = np.float32(ev)
                elif v.dtype == np.float64:
                    ev = np.float64(ev)

                self.assertEqual(ev, v)


        # test that numpy with float64 raises exception
        toy_example = ToyExample(feature_bool = feature_bool,
                                 feature_int32 = feature_int32,
                                 feature_int64 = feature_int64,
                                 feature_float = feature_float,
                                 feature_double = feature_double,
                                 feature_string = feature_string,
                                 feature_np_array_int32 = np.ones(shape, dtype=np.int64), # int64
                                 feature_np_array_float32 = np.random.random(shape) # float64
                                 )
        self.assertRaises(TypeError, toy_example.serialize_to_string)


class ToyExample(Example):
    """
    This class is used to check that all attribute types are correctly handled.
    """

    def __init__(self,
                 feature_bool,
                 feature_int32,
                 feature_int64,
                 feature_float,
                 feature_double,
                 feature_string,
                 feature_np_array_int32,
                 feature_np_array_float32):

        super(ToyExample, self).__init__()
        self._feature_bool = feature_bool
        self._feature_int32 = feature_int32
        self._feature_int64 = feature_int64
        self._feature_float = feature_float
        self._feature_double = feature_double
        self._feature_string = feature_string
        self._feature_np_array_int32 = feature_np_array_int32
        self._feature_np_array_float32 = feature_np_array_float32

    @tfrecordable(dtype=Example.Field.TYPE_BOOL)
    def feature_bool(self):
        return self._feature_bool

    @feature_bool.setter
    def feature_bool(self, val):
        self._feature_bool = val

    @tfrecordable(dtype=Example.Field.TYPE_INT32)
    def feature_int32(self):
        return self._feature_int32

    @feature_int32.setter
    def feature_int32(self, val):
        self._feature_int32 = val

    @tfrecordable(dtype=Example.Field.TYPE_INT64)
    def feature_int64(self):
        return self._feature_int64

    @feature_int64.setter
    def feature_int64(self, val):
        self._feature_int64 = val

    @tfrecordable(dtype=Example.Field.TYPE_FLOAT)
    def feature_float(self):
        return self._feature_float

    @feature_float.setter
    def feature_float(self, val):
        self._feature_float = val

    @tfrecordable(dtype=Example.Field.TYPE_DOUBLE)
    def feature_double(self):
        return self._feature_double

    @feature_double.setter
    def feature_double(self, val):
        self._feature_double = val

    @tfrecordable(dtype=Example.Field.TYPE_STRING)
    def feature_string(self):
        return self._feature_string

    @feature_string.setter
    def feature_string(self, val):
        self._feature_string = val

    @tfrecordable(dtype=Example.Field.TYPE_ARRAY_INT32)
    def feature_np_array_int32(self):
        return self._feature_np_array_int32

    @feature_np_array_int32.setter
    def feature_np_array_int32(self, val):
        self._feature_np_array_int32 = val

    @tfrecordable(dtype=Example.Field.TYPE_ARRAY_FLOAT32)
    def feature_np_array_float32(self):
        return self._feature_np_array_float32

    @feature_np_array_float32.setter
    def feature_np_array_float32(self, val):
        self._feature_np_array_float32 = val
        
    def load(self, **kwargs):
        pass # unused

    def to_csv_row(self):
        pass # unused

    @classmethod
    def from_csv_row(cls, row, **kwargs):
        pass # unused


        
        
if __name__ == '__main__':
    unittest.main()
