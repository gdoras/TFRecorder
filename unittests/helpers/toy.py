import numpy as np
import os
import random
from unittests.helpers.toy_example_1 import ToyExample1
from unittests.helpers.toy_example_2 import ToyExample2

import tfrecorder.helpers.utils as utils


def generate_toy_examples(save_directory_path,
                          example_class,
                          num_examples=10,
                          data_shape=(128, 4)):



    if not os.path.exists(save_directory_path):
        os.mkdir(save_directory_path)

    uplets = []

    if example_class is ToyExample1:

        num_labels = 5

        for i in range(num_examples):

            name = utils.generate_random_string() # 6 chars
            data_filepath = os.path.join(save_directory_path, '%s.npy' % name)
            np.save(data_filepath, np.random.random(data_shape).astype(np.float32))

            label = random.randint(0, num_labels)
            likelihood = np.float32(random.random())
            uplets.append((name, label, likelihood, data_filepath))

    elif example_class is ToyExample2:

        src_data_directory_path = os.path.join(save_directory_path, 'src')
        os.mkdir(src_data_directory_path)
        tgt_data_directory_path = os.path.join(save_directory_path, 'tgt')
        os.mkdir(tgt_data_directory_path)

        for i in range(num_examples):

            name = utils.generate_random_string() # 6 chars
            src_data_filepath = os.path.join(src_data_directory_path, '%s.npy' % name)
            np.save(src_data_filepath, np.random.random(data_shape).astype(np.float32))

            tgt_data_filepath = os.path.join(tgt_data_directory_path, '%s.npy' % name)
            np.save(tgt_data_filepath, np.random.random(data_shape).astype(np.float32))

            uplets.append((str(i), name))

    else:
        raise ValueError('Example class %s is not supported.' % example_class)



    examples = [example_class(*u) for u in uplets]

    return examples


