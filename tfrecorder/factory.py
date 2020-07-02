import os
import glob
import random
import tensorflow as tf

import tfrecorder.helpers.engine as engine
import tfrecorder.helpers.constants as cts
import tfrecorder.config.parser as config_parser
import tfrecorder.helpers.utils as utils

# wrappers to deal with various use cases

def generate_and_save_tfrecords_files_for_examples_file(save_directory_path,
                                                        example_class,
                                                        examples_filepath,
                                                        examples_tfrecord_file_max_size_in_bytes=1e6,
                                                        examples_log_in_csv_file=True,
                                                        **kwargs
                                                        ):
    """
    Simple wrapper around generate_and_save_tfrecords_files_for_examples.

    Args:
        save_directory_path: str, path of the directory where to save the various directories
        example_class: class object, the Example subclass to use
        examples_filepath: str, path to a csv file that describes the examples to instantiate and to use
        examples_tfrecord_file_max_size_in_bytes: int, maximum size in bytes of a tfrecord file.
        examples_log_in_csv_file: bool, whether to log the metadata of the examples in a csv file.

    """
    examples = example_class.from_csv_file(examples_filepath,
                                           **kwargs)

    engine.generate_and_save_tfrecords_files_for_examples(save_directory_path,
                                                          examples,
                                                          examples_tfrecord_file_max_size_in_bytes=examples_tfrecord_file_max_size_in_bytes,
                                                          examples_log_in_csv_file=examples_log_in_csv_file,
                                                          **kwargs
                                                          )


def generate_and_save_train_eval_test_tfrecords_files(save_directory_path,
                                                      examples_class,
                                                      examples_filepath,
                                                      examples_train_eval_test_ratio=0.8,
                                                      examples_tfrecord_file_max_size_in_bytes=1e6,
                                                      examples_log_in_csv_file=True,
                                                      **kwargs
                                                      ):
    """
    Simple wrapper around generate_and_save_train_eval_test_tfrecords_files_for_examples.

    Args:
        save_directory_path: str, path of the directory where to save the various directories
        examples_class: class object, the Example subclass to use
        examples_filepath: str, path to a csv file that describes the examples to instantiate and to use.
        examples_train_eval_test_ratio: float, or list, ratios to split the Example objects list.
        examples_tfrecord_file_max_size_in_bytes: int, maximum size in bytes of a tfrecord file.
        examples_log_in_csv_file: bool, whether to log the metadata of the examples in a csv file.

    """

    examples = examples_class.from_csv_file(examples_filepath,
                                            **kwargs)

    counts = generate_and_save_train_eval_test_tfrecords_files_for_examples(save_directory_path,
                                                                            examples,
                                                                            examples_train_eval_test_ratio=examples_train_eval_test_ratio,
                                                                            examples_tfrecord_file_max_size_in_bytes=examples_tfrecord_file_max_size_in_bytes,
                                                                            examples_log_in_csv_file=examples_log_in_csv_file,
                                                                            **kwargs
                                                                            )

    # save a config file
    config_filepath = os.path.join(save_directory_path, cts.EXAMPLES_TFRECORD_FILES_CONFIG_FILENAME)
    config_parser.save_config(config_filepath,
                              examples_class,
                              examples_filepath,
                              examples_train_eval_test_ratio,
                              examples_tfrecord_file_max_size_in_bytes,
                              examples_log_in_csv_file,
                              counts,
                              **kwargs)


def generate_and_save_tfrecords_files_for_examples_dict(save_directory_path,
                                                        examples_dict,
                                                        examples_tfrecord_file_max_size_in_bytes=1e6,
                                                        examples_log_in_csv_file=True,
                                                        **kwargs
                                                        ):
    """
    Creates the directories (e.g. train/eval or fold_0, fold_1, ...) and stores the examples in tfrecords files for
    each directory.

    Args:
        save_directory_path: str, path of the directory where to save the various directories
        examples_dict: dict, of the form {dirname: list of Example objects}
        examples_tfrecord_file_max_size_in_bytes: int, maximum size in bytes of a tfrecord file.
        examples_log_in_csv_file: bool, whether to log the metadata of the examples in a csv file.
        **kwargs: dict, arguments for the Example subclass methods such as load, etc.

    """

    if not os.path.exists(save_directory_path):
        os.mkdir(save_directory_path)

    # just to make sure that logfile is at the root dir, not in the first subdir
    logger = utils.get_logger(name=engine.LOGGER_NAME,
                              logs_directory_path=save_directory_path)

    counts_dict = {}
    for subdir_name, examples in examples_dict.items():

        subdir_path = os.path.join(save_directory_path, subdir_name)

        engine.generate_and_save_tfrecords_files_for_examples(subdir_path,
                                                              examples,
                                                              examples_tfrecord_file_max_size_in_bytes=examples_tfrecord_file_max_size_in_bytes,
                                                              examples_log_in_csv_file=examples_log_in_csv_file,
                                                              **kwargs
                                                              )

        counts_dict[subdir_name] = len(examples)

    return counts_dict



def generate_and_save_tfrecords_files_for_train_eval_test_examples_files(save_directory_path,
                                                                         example_class,
                                                                         train_examples_filepath,
                                                                         eval_examples_filepath,
                                                                         test_examples_filepath=None,
                                                                         examples_tfrecord_file_max_size_in_bytes=1e6,
                                                                         examples_log_in_csv_file=True,
                                                                         **kwargs
                                                                         ):
    """
    Split the examples into train and eval (optionally test) sets, creates corresponding directories and store the
    examples in tfrecords files for each set.

    Args:
        save_directory_path: str, path of the directory where to save the various directories
        example_class: class object, the Example subclass to use
        train_examples_filepath: str, path to a csv file that describes the examples to instantiate and to use for train
        eval_examples_filepath: str, path to a csv file that describes the examples to instantiate and to use for eval
        test_examples_filepath: str, path to a csv file that describes the examples to instantiate and to use for test, or None.
        examples_tfrecord_file_max_size_in_bytes: int, maximum size in bytes of a tfrecord file.
        examples_log_in_csv_file: bool, whether to log the metadata of the examples in a csv file.

    """

    examples_filepaths_dict = {cts.TRAIN_DIRECTORY_NAME: train_examples_filepath,
                               cts.EVAL_DIRECTORY_NAME: eval_examples_filepath}

    if test_examples_filepath:
        examples_filepaths_dict[cts.TEST_DIRECTORY_NAME] = test_examples_filepath

    return generate_and_save_tfrecords_files_for_examples_files_dict(save_directory_path,
                                                                     example_class,
                                                                     examples_filepaths_dict,
                                                                     examples_tfrecord_file_max_size_in_bytes=examples_tfrecord_file_max_size_in_bytes,
                                                                     examples_log_in_csv_file=examples_log_in_csv_file,
                                                                     **kwargs
                                                                     )


def generate_and_save_train_eval_test_tfrecords_files_for_examples(save_directory_path,
                                                                   examples,
                                                                   examples_train_eval_test_ratio=0.8,
                                                                   examples_shuffle=True,
                                                                   examples_tfrecord_file_max_size_in_bytes=1e6,
                                                                   examples_log_in_csv_file=True,
                                                                   **kwargs
                                                                   ):
    """
    Split the examples into train and eval (optionally test) sets, creates corresponding directories and store the
    examples in tfrecords files for each set.

    Args:
        save_directory_path: str, path of the directory where to save the various directories
        examples: list, of Example objects
        examples_train_eval_test_ratio: float, or list, ratios to split the Example objects list.
        examples_shuffle: bool, whether to shuffle examples before split.
        examples_tfrecord_file_max_size_in_bytes: int, maximum size in bytes of a tfrecord file.
        examples_log_in_csv_file: bool, whether to log the metadata of the examples in a csv file.

    """
    if examples_shuffle:
        random.shuffle(examples)

    train_examples, eval_examples, test_examples = split_examples_list(examples, examples_train_eval_test_ratio)

    examples_dict = {cts.TRAIN_DIRECTORY_NAME: train_examples,
                     cts.EVAL_DIRECTORY_NAME: eval_examples}

    if test_examples:
        examples_dict[cts.TEST_DIRECTORY_NAME] = test_examples

    return generate_and_save_tfrecords_files_for_examples_dict(save_directory_path,
                                                               examples_dict,
                                                               examples_tfrecord_file_max_size_in_bytes=examples_tfrecord_file_max_size_in_bytes,
                                                               examples_log_in_csv_file=examples_log_in_csv_file,
                                                               **kwargs)


def generate_and_save_tfrecords_files_for_examples_files_dict(save_directory_path,
                                                              example_class,
                                                              examples_filepaths_dict,
                                                              examples_tfrecord_file_max_size_in_bytes=1e6,
                                                              examples_log_in_csv_file=True,
                                                              **kwargs
                                                              ):
    """
    Creates the directories (e.g. train/eval or fold_0, fold_1, ...) and stores the examples in tfrecords files for
    each directory.

    Args:
        save_directory_path: str, path of the directory where to save the various directories
        example_class: class object, the Example subclass to use
        examples_filepaths_dict: dict, of the form {dirname: filepath to a csv file that describes the examples to
                                instantiate and to use for eval}
        examples_tfrecord_file_max_size_in_bytes: int, maximum size in bytes of a tfrecord file.
        examples_log_in_csv_file: bool, whether to log the metadata of the examples in a csv file.

    Returns:
        counts: dict, the names of the directories and the number of examples in each.

    """

    if not os.path.exists(save_directory_path):
        os.mkdir(save_directory_path)

    examples_dict = {subdir_name: example_class.from_csv_file(examples_filepath,
                                                              **kwargs) for subdir_name, examples_filepath in examples_filepaths_dict.items()}

    return generate_and_save_tfrecords_files_for_examples_dict(save_directory_path,
                                                               examples_dict,
                                                               examples_tfrecord_file_max_size_in_bytes=examples_tfrecord_file_max_size_in_bytes,
                                                               examples_log_in_csv_file=examples_log_in_csv_file,
                                                               **kwargs
                                                               )


# dataset

def generate_dataset(tfrecords_filepaths,
                     example_class,
                     dataset_num_shuffled_tfrecord_files=None,
                     dataset_fetching_num_threads=None
                     ):
    """
    Generate a dataset with a list of tfrecords filepaths. The dataset instantiate the protobuf for the given Example
    subclass.
    By default, it does not batch nor shuffle the results so that order is preserved and consistency between original data and
    stored data can be checked.

    Args:
        tfrecords_filepaths: list,
        example_class: class, of Example subclass
        dataset_num_shuffled_tfrecord_files: int, size of the tfrecords filepaths buffer to shuffle before deserializing.
        dataset_fetching_num_threads: int


    Returns:

    """
    # create one dataset object out of the list of tfrecords files
    dataset = tf.data.TFRecordDataset(tfrecords_filepaths)

    if dataset_num_shuffled_tfrecord_files:
        dataset = dataset.shuffle(buffer_size=dataset_num_shuffled_tfrecord_files)

    # parse the records into tensors describing sources
    dataset = dataset.map(lambda serialized_examples : example_class.parse_from_string(serialized_examples),
                          num_parallel_calls=dataset_fetching_num_threads)


    return dataset


# utils

def get_tfrecord_filepaths(dirpath):
    """
    Convenience function to get a reference to the list of records files in this dir path.
    Args:
        dirpath: str, the dirpath

    Returns:
        filepath: str, the filepath
    """
    if not os.path.exists(dirpath) or not os.path.isdir(dirpath):
        raise ValueError('There is no directory at %s.' % dirpath)

    # make sure we sort the tfrecords in the same order than they were created. This can be shuffled afterward.
    tfrecord_filepaths = sorted(glob.glob(os.path.join(dirpath, '*.tfr')),
                                key= lambda fp: int(os.path.splitext(os.path.basename(fp))[0]))


    return tfrecord_filepaths


def get_examples_list_filepaths(dirpath):
    """
    Convenience function to get a reference to the examples.csv files in this dir path.
    Args:
        dirpath: str, the dirpath

    Returns:
        filepath: str, the filepath
    """
    if not os.path.exists(dirpath) or not os.path.isdir(dirpath):
        raise ValueError('There is no directory at %s.' % dirpath)

    return os.path.join(dirpath, cts.EXAMPLES_LIST_FILENAME)


def split_examples_list(examples, examples_train_eval_test_ratio):

    num_examples = len(examples)

    if type(examples_train_eval_test_ratio) is list:
        examples_train_eval_test_ratios = examples_train_eval_test_ratio
    else:
        examples_train_eval_test_ratios = [examples_train_eval_test_ratio]

    if len(examples_train_eval_test_ratios) == 1:
        num_train_examples = int(num_examples*examples_train_eval_test_ratios[0])
        num_eval_examples = num_examples - num_train_examples
        num_test_examples = 0

    elif len(examples_train_eval_test_ratios) == 2:

        if sum(examples_train_eval_test_ratios) != 1.0:
            raise ValueError('Train, eval ratio shall sum up to 1.0 (found %.1f).' % sum(examples_train_eval_test_ratios))

        num_train_examples = int(num_examples*examples_train_eval_test_ratios[0])
        num_eval_examples = num_examples - num_train_examples
        num_test_examples = 0

    elif len(examples_train_eval_test_ratios) == 3:

        if sum(examples_train_eval_test_ratios) != 1.0:
            raise ValueError('Train, eval, test ratio shall sum up to 1.0 (found %.1f).' % sum(examples_train_eval_test_ratios))

        num_train_examples = int(num_examples*examples_train_eval_test_ratios[0])
        num_eval_examples = int(num_examples*examples_train_eval_test_ratios[1])
        num_test_examples = num_examples - num_train_examples - num_eval_examples

    else:
        raise ValueError('There should be only 3 ratios for train, test and eval.')


    if num_test_examples > 0:
        return examples[:num_train_examples], \
               examples[num_train_examples:num_train_examples+num_eval_examples], \
               examples[num_train_examples+num_eval_examples:]
    else:
        return examples[:num_train_examples], \
               examples[num_train_examples:num_train_examples+num_eval_examples], \
               None