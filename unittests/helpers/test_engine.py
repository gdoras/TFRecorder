import unittest
import tempfile
import os

import tfrecorder.helpers.checker as checker
import unittests.helpers.toy as toy
from unittests.helpers.toy_example_1 import ToyExample1
from unittests.helpers.toy_example_2 import ToyExample2

class EngineTestCase(unittest.TestCase):



    def test_generate_and_save_tfrecords_files_for_examples(self):
        """
        Here we create the tfrecords for a list of examples. This is the basic functionality that is used by all other
        scenarios which are simply wrappers boiling down to this one.
        """

        example_classes = [ToyExample1, ToyExample2]
        for example_class in example_classes:

            print('Testing %s...' % example_class.__name__)

            num_examples = 87
            data_shape=[37, 3]

            # test with and without arguments passed to the Example.load(**kwargs) method
            various_kwargs = [{}, dict(add_row = True)]
            for kwargs in various_kwargs:

                with tempfile.TemporaryDirectory() as save_directory_path:

                    corpus_directory_path = os.path.join(save_directory_path, 'corpus')
                    examples = toy.generate_toy_examples(corpus_directory_path,
                                                         example_class,
                                                         num_examples=num_examples,
                                                         data_shape=data_shape)

                    # kwargs must include the kwargs required by `to_csv_file`, `load` and `split` methods,
                    # when applicable
                    if example_class is ToyExample1:
                        kwargs.update(dict(data_dirpath = corpus_directory_path))

                    elif example_class is ToyExample2:

                        kwargs.update(dict(src_data_dirpath = os.path.join(corpus_directory_path, 'src'),
                                           tgt_data_dirpath = os.path.join(corpus_directory_path, 'tgt'),
                                           chunk_size_in_bins = 5))

                    # we moved the assertion logic in the checker
                    res = checker.assert_example_serialize_deserialize_is_ok(examples,
                                                                             tfrecords_files_max_size_in_bytes=1e4,
                                                                             **kwargs)


                    self.assertTrue(res)




if __name__ == '__main__':
    unittest.main()
