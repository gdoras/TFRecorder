import tensorflow as tf
import os
import time
import csv
from tqdm import tqdm

import tfrecorder.helpers.constants as cts
import tfrecorder.helpers.utils as utils
from tfrecorder.helpers.marshaller import IgnoreExampleException

LOGGER_NAME = 'TFRecorder'

def generate_and_save_tfrecords_files_for_examples(save_directory_path,
                                                   examples,
                                                   examples_tfrecords_files_max_size_in_bytes=1e6,
                                                   examples_log_in_csv_file=True,
                                                   progress_bar=True,
                                                   **kwargs):
    """
    This is the core of the TFRecorder logic.
    This function instantiates, pre-processes and stacks the examples data into tfrecords.
    Once a given tfrecord file has reached the maximum size, it is saved, and a new tfrecord files is started.

    Args:
        save_directory_path: str, where to save the tfrecord files and other related files.
        examples: list, of Example objects
        examples_tfrecords_files_max_size_in_bytes: int, maximum size in bytes of a tfrecord file.
        examples_log_in_csv_file: bool, whether to log the metadata of the examples in a csv file.
        progress_bar: bool, display a progress bar.

    Returns:
        -

    """

    if not os.path.exists(save_directory_path):
        os.mkdir(save_directory_path)

    logger = utils.get_logger(name=LOGGER_NAME,
                              logs_directory_path=save_directory_path) # if already set, logs_dir is not taken into account.

    # stack all classes samples data into tfrecords files
    num_examples = len(examples)
    tag = os.path.basename(save_directory_path)
    logger.info('Saving %s tfrecords files for %d examples...' % (tag, num_examples))

    if examples_log_in_csv_file:
        examples_list_filepath = os.path.join(save_directory_path, cts.EXAMPLES_LIST_FILENAME)
        logger.info('   Metadata of the examples will be saved in a csv file.')
    else:
        examples_list_filepath = None
        logger.info('   No metadata of the examples will be saved in a csv file.')

    start_time = time.time()

    current_tfrecord_file_count = 0
    current_tfrecord_file_content_size_in_bytes = 0

    # as we dont want to keep data in memory, we write it to tfrecord right after it has been loaded.
    current_tfrecord_filepath = os.path.join(save_directory_path, '%s.tfr' % current_tfrecord_file_count)
    current_tfrecord_file_writer = tf.io.TFRecordWriter(current_tfrecord_filepath)

    i, j = -1, 0
    if progress_bar:
        examples = tqdm(examples)

    for i, example in enumerate(examples):

        # instantiate the data of this example
        try:
            example.load(**kwargs)
        except IgnoreExampleException as e:
            logger.warning(e)
            continue # ignore this example

        # optionally, log the metadata of this example (before it is optionally chunked)
        if examples_log_in_csv_file:
            with open(examples_list_filepath, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(example.to_csv_row())

        # in case this example's data needs to be chunked. If not, simply returns a list containing this single example.
        try:
            chunked_examples = example.split(**kwargs)
        except IgnoreExampleException as e:
            logger.warning(e)
            continue # ignore this example

        # if we have split the original example, no need to keep it around
        if chunked_examples is not None:
            example.release()
            count_chunks = True
        else:
            chunked_examples = [example]
            count_chunks = False

        for chunked_example in chunked_examples:

            # now we can check the full size of the example that will be stored
            chunked_example_size = chunked_example.get_byte_size()

            # we will stack the data until they reach a memory limit
            if current_tfrecord_file_content_size_in_bytes + chunked_example_size <= examples_tfrecords_files_max_size_in_bytes:

                # we haven't reach the max yet, keep adding to the current file
                #with tf.io.TFRecordWriter(current_tfrecord_filepath) as writer:
                current_tfrecord_file_writer.write(chunked_example.serialize_to_string())
                current_tfrecord_file_writer.flush()
                current_tfrecord_file_content_size_in_bytes += chunked_example_size

            else:

                # previous file has reach its max size, create another one.
                current_tfrecord_file_writer.close()
                current_tfrecord_file_count += 1

                current_tfrecord_filepath = os.path.join(save_directory_path, '%s.tfr' % current_tfrecord_file_count)
                current_tfrecord_file_writer = tf.io.TFRecordWriter(current_tfrecord_filepath)

                #with tf.io.TFRecordWriter(current_tfrecord_filepath) as writer:
                current_tfrecord_file_writer.write(chunked_example.serialize_to_string())
                current_tfrecord_file_writer.flush()
                current_tfrecord_file_content_size_in_bytes = chunked_example_size

            chunked_example.release()

        if count_chunks:
            j += len(chunked_examples)

        del chunked_examples

        hop = max(10, num_examples // 10)
        if not progress_bar and i > 0  and (i+1) % hop == 0:

            num_chunked_examples = ' (%d chunks)' % j if j > 0 else ''
            logger.info("   Processed %d / %d examples%s and saved %d tfrecords files (eta: %s)..." %
                        (i+1,
                         num_examples,
                         num_chunked_examples,
                         current_tfrecord_file_count,
                         utils.eta_based_on_elapsed_time(i, num_examples, start_time)))

        # force clean up now
        #gc.collect()

    current_tfrecord_file_writer.close()
    current_tfrecord_file_count += 1

    num_chunked_examples = ' (%d chunks)' % j if j > 0 else ''
    logger.info("   Processed %d / %d examples%s and saved %d tfrecords files." %
                (i+1,
                 num_examples,
                 num_chunked_examples,
                 current_tfrecord_file_count))