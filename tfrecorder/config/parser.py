from configparser import ConfigParser
import os
import json


def save_config(config_filepath,
                examples_class,
                examples_filepath,
                examples_train_eval_test_ratio,
                examples_tfrecord_file_max_size_in_bytes,
                examples_log_in_csv_file,
                counts,
                **kwargs):
    """
    Save a config file with these parameters.

    Args:
        config_filepath:
        examples_class:
        examples_filepath:
        examples_train_eval_test_ratio:
        examples_tfrecord_file_max_size_in_bytes:
        examples_log_in_csv_file:
        counts:
        **kwargs:

    Returns:

    """
    # kwargs first
    examples_dict = {
        'examples' : {

        }
    }

    # make sure we save list as json list
    for k, v in kwargs.items():
        if type(v) is list:
            kwargs[k] = json.dumps(v)

    # remove the examples_ prefix
    kwargs = {k.replace('examples_', ''): v for k, v in kwargs.items() }
    # do a bit of reformating
    kwargs = {k: os.path.abspath(v) if 'path' in k else str(v) for k, v in kwargs.items() }
    examples_dict['examples'].update(kwargs)

    # basics
    examples_dict['examples'].update(
        {
            'class_name': examples_class.__name__,
            'list_filepath': os.path.abspath(examples_filepath),
            'tfrecord_file_max_size_in_mb': '%.2e' % (examples_tfrecord_file_max_size_in_bytes / 1e6),
            'log_in_csv_file': str(examples_log_in_csv_file),
        }
    )
    if examples_train_eval_test_ratio:
        examples_dict['examples'].update(
            {
                'train_eval_test_ratio': str(examples_train_eval_test_ratio),
            })

    # counts
    counts = {'%s_count' % k: str(v) for k, v in counts.items()}
    examples_dict['examples'].update(counts)



    # create config
    config = ConfigParser()
    config.read_dict(examples_dict)

    with open(config_filepath, 'w') as configfile:
        config.write(configfile)


