import tensorflow as tf
import numpy as np
import os
import tempfile

import tfrecorder.factory as tf_factory
from tfrecorder.helpers.marshaller import Example
import tfrecorder.helpers.constants as cts
import tfrecorder.helpers.utils as utils

LOGGER_NAME = 'TFRecorderChecker'

def assert_example_serialize_deserialize_is_ok(example,
                                               tfrecords_files_max_size_in_bytes=1e6,
                                               **kwargs):
    """
    Check that this example can be serialized and deserialized correctly.

    Args:
        example: an Example, or a list of Example, object(s).
        tfrecords_files_max_size_in_bytes: int
        kwargs: dict, must include the kwargs required by `to_csv_file`, `load` and `split` methods.

    """
    logger = utils.get_logger()

    if type(example) is list:
        examples = example
    else:
        examples = [example]

    example_class = examples[0].__class__

    logger.info('Checking consistency of %d instances of class %s...' % (len(examples),
                                                                         example_class.__name__))

    # create tfrecord for this example
    with tempfile.TemporaryDirectory() as tmp_directory_path:

        # create the csv file
        logger.info('Saving examples metadata to csv file...')
        csv_directory_path = os.path.join(tmp_directory_path, 'examples')
        os.mkdir(csv_directory_path)
        examples_list_filepath = os.path.join(csv_directory_path, cts.EXAMPLES_LIST_FILENAME)
        Example.to_csv_file(examples_list_filepath, examples)

        # create the tfrecords
        logger.info('Serializing examples data to tfrecords files...')
        tfr_directory_path = os.path.join(tmp_directory_path, 'checker')
        os.mkdir(tfr_directory_path)

        tf_factory.generate_and_save_tfrecords_files_for_examples_file(tfr_directory_path,
                                                                       example_class,
                                                                       examples_list_filepath,
                                                                       tfrecords_files_max_size_in_bytes=tfrecords_files_max_size_in_bytes,
                                                                       log_in_csv_file=True,
                                                                       **kwargs
                                                                       )

        # check that we have stored a csv examples list if the Example implemented the to_csv_row method
        logger.info('Checking that examples metadata has been saved...')
        examples_list_filepath = os.path.join(tfr_directory_path, cts.EXAMPLES_LIST_FILENAME)
        assert os.path.exists(examples_list_filepath)


        # check that the tfrecords exist
        logger.info('Checking that tfrecords files have been created...')
        tfrecord_filepaths = tf_factory.get_tfrecord_filepaths(tfr_directory_path)
        for tfrecord_filepath in tfrecord_filepaths:
            assert os.path.exists(tfrecord_filepath)


        # generate a dataset and check that the data stored is correct
        logger.info('Building a tf.data.Dataset with the tfrecords files...')
        dataset_batch_size = 0 # dont batch
        dataset = generate_test_dataset(tfrecord_filepaths,
                                        example_class,
                                        dataset_batch_size=dataset_batch_size)

        # check that the @tfrecordable attributes have been stored and are correct
        assert_examples_content_matches_dataset_content(examples,
                                                       dataset,
                                                        **kwargs)

        logger.info('Consistency of %d instances of class %s checked.\n' % (len(examples),
                                                                               example_class.__name__))

        return True



def assert_examples_content_matches_dataset_content(examples,
                                                    dataset,
                                                    **kwargs):
    """
    This tests that the content of the examples and the content of the examples stored in the dataset are the same.
    This is a generic test that does not depend on the exact Example subclass.
    """

    logger = utils.get_logger()
    num_examples = len(examples)

    # check that the @tfrecordable attributes have been stored, are in the correct order and have correct values.
    ks = examples[0].get_tfrecordable_attribute_names()

    logger.info('Checking consistency of %d attributes %s for first example (out of %d)...' % (len(ks),
                                                                                               ' - '.join(ks),
                                                                                               num_examples))

    logged_first = False

    def yield_example_chunk_data(exs, **kw):
        for e in exs:
            e.load(**kw)
            ces = e.split(**kw)

            for ce in ces:
                yield ce

    chunk_generator = yield_example_chunk_data(examples, **kwargs)
    for chunked_example, value_tensors in zip(chunk_generator, dataset):

        for k, vt in zip(ks, value_tensors):

            ev = getattr(chunked_example, k) # example original values
            mv = vt.numpy()          # protobuf message values to numpy

            if not logged_first:
                logger.info('   Checking attribute %s...' % k)

            if type(ev) is np.ndarray:

                # protobuf is a flat array, we reshape it to the expected size
                mv = mv.reshape(ev.shape)
                if not np.array_equal(ev, mv):
                    raise AssertionError('Attribute %s has different values in example and protobuf (%s vs. %s).' % (k, ev, mv))

            elif type(ev) is str:
                mv = mv.decode("utf-8")
                if ev != mv:
                    raise AssertionError('Attribute %s has different values in example and protobuf (%s vs. %s).' % (k, ev, mv))

            else:
                if mv.dtype == np.float32:
                    ev = np.float32(ev)
                elif mv.dtype == np.float64:
                    ev = np.float64(ev)

                if ev != mv:
                    raise AssertionError('Attribute %s has different values in example and protobuf (%s vs. %s).' % (k, ev, mv))

        if not logged_first:
            logger.info('Consistency of %d attributes %s for first example checked.\n'
                        'Checking %d subsequent examples...' % (len(ks),' - '.join(ks), num_examples-1))
        logged_first = True


    logger.info('Done.')

def generate_test_dataset(tfrecords_filepaths,
                         example_class,
                         dataset_batch_size=0,
                         dataset_batch_drop_remainder=False,
                         dataset_fetching_num_threads=1
                         ):
    """
    Generate a dataset with a list of tfrecords filepaths. The dataset instantiate the protobuf for the given Example
    subclass indicated by `example_class_module_name` and `example_class_module_name`.
    By default, it does not batch nor shuffle the results so that order is preserved and consistency between original data and
    stored data can be checked.

    Args:
        tfrecords_filepaths: list,
        example_class: class, of Example subclass
        dataset_batch_size: int, batch size.
        dataset_batch_drop_remainder: bool, if batch > 0, whether to drop or not incomplete batches
        dataset_fetching_num_threads: int


    Returns:

    """
    # create one dataset object out of the list of tfrecords files
    dataset = tf_factory.generate_dataset(tfrecords_filepaths,
                                  example_class,
                                  dataset_fetching_num_threads=dataset_fetching_num_threads)

    if dataset_batch_size > 0:
        dataset = dataset.batch(dataset_batch_size, drop_remainder=dataset_batch_drop_remainder)

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return dataset