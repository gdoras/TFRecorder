import unittest
import tempfile
import os

import tfrecorder.factory as tf_factory
import tfrecorder.helpers.constants as cts
from tfrecorder.helpers.marshaller import Example
import unittests.helpers.toy as toy
from unittests.helpers.toy_example_1 import ToyExample1
from unittests.helpers.toy_example_2 import ToyExample2


class FactoryTestCase(unittest.TestCase):




    def test_generate_and_save_tfrecords_files_for_examples_dict(self):
        """
        Here we create the tfrecords for a list of examples that is split between different sets.
        This will be used e.g. for train/eval/test sets, or fold_0/fold_1/etc. sets for k-folding.
        """

        example_classes = [ToyExample1, ToyExample2]

        for example_class in example_classes:

            with tempfile.TemporaryDirectory() as tmp_directory_path:

                num_examples = 13

                corpus_directory_path = os.path.join(tmp_directory_path, 'corpus')
                examples = toy.generate_toy_examples(corpus_directory_path,
                                                     example_class=example_class,
                                                     num_examples=num_examples)

                kwargs = {}
                if example_class is ToyExample1:
                    kwargs.update(dict(data_dirpath = corpus_directory_path))

                elif example_class is ToyExample2:

                    kwargs.update(dict(src_data_dirpath = os.path.join(corpus_directory_path, 'src'),
                                       tgt_data_dirpath = os.path.join(corpus_directory_path, 'tgt'),
                                       chunk_size_in_bins = 5))


                save_directory_path = os.path.join(tmp_directory_path, 'tfrecords')
                examples_dict = {'train': examples[:-2],
                                 'eval': examples[-2:]
                                 }


                tf_factory.generate_and_save_tfrecords_files_for_examples_dict(save_directory_path,
                                                                               examples_dict,
                                                                               **kwargs)


                for subdir_name in examples_dict.keys():

                    # check that we have created each subdir and that each subdir contains the tfr files.
                    subdir_path = os.path.join(save_directory_path, subdir_name)
                    self.assertTrue(os.path.exists(subdir_path))
                    for tfrecord_filepath in tf_factory.get_tfrecord_filepaths(subdir_path):
                        self.assertTrue(os.path.exists(tfrecord_filepath))



    def test_generate_and_save_tfrecords_files_for_train_eval_test_examples_files(self):
        """
        Here we create a first splits of train/eval/test tfrecords sets.
        We then use the examples files to recreate a second train/eval/test tfrecords sets.

        This test the case where we want to recreate a train/eval/test sets from the metadata that was used in other
        experiments for instance.

        """

        example_classes = [ToyExample1, ToyExample2]

        for example_class in example_classes:

            with tempfile.TemporaryDirectory() as tmp_directory_path:

                num_examples = 13
                ratios = [0.5, 0.3, 0.2]
                corpus_directory_path = os.path.join(tmp_directory_path, 'corpus')
                examples = toy.generate_toy_examples(corpus_directory_path,
                                                     example_class=example_class,
                                                     num_examples=num_examples)

                kwargs = {}
                if example_class is ToyExample1:
                    kwargs.update(dict(data_dirpath = corpus_directory_path))

                elif example_class is ToyExample2:

                    kwargs.update(dict(src_data_dirpath = os.path.join(corpus_directory_path, 'src'),
                                       tgt_data_dirpath = os.path.join(corpus_directory_path, 'tgt'),
                                       chunk_size_in_bins = 5))

                # create a first split so that we can use the examples.csv files that will describe train/eval/test sets.
                save_directory_path = os.path.join(tmp_directory_path, 'tfrecords_original')

                tf_factory.generate_and_save_train_eval_test_tfrecords_files_for_examples(save_directory_path,
                                                                                          examples,
                                                                                          ratios,
                                                                                          **kwargs)

                # check that we have the examples.csv files. This should be the case as ToyExample implements to_csv_row.
                self._test_train_eval_test_sets_directories(save_directory_path)


                # now we create a second set from the files of the original sets
                subdir_names = [cts.TRAIN_DIRECTORY_NAME, cts.EVAL_DIRECTORY_NAME, cts.TEST_DIRECTORY_NAME]
                examples_list_filepaths = [tf_factory.get_examples_list_filepaths(os.path.join(save_directory_path, subdir_name)) for subdir_name in subdir_names]

                save_directory_path = os.path.join(tmp_directory_path, 'tfrecords_recreated')

                train_examples_filepath, \
                eval_examples_filepath, \
                test_examples_filepath = examples_list_filepaths

                kwargs.update({'data_dirpath': corpus_directory_path,
                               'add_row': True})

                tf_factory.generate_and_save_tfrecords_files_for_train_eval_test_examples_files(save_directory_path,
                                                                                                example_class,
                                                                                                train_examples_filepath,
                                                                                                eval_examples_filepath,
                                                                                                test_examples_filepath,
                                                                                                **kwargs)

                # check that we have what was expected
                self._test_train_eval_test_sets_directories(save_directory_path)



    def test_generate_and_save_train_eval_test_tfrecords_files_for_examples(self):
        """
        Here we create a various train/eval/test tfrecords sets for various ratios.
        """
        example_classes = [ToyExample1, ToyExample2]

        for example_class in example_classes:

            with tempfile.TemporaryDirectory() as tmp_directory_path:

                num_examples = 13
                various_ratios = [0.8,
                                  [0.7, 0.3],
                                  [0.5, 0.3, 0.2]]
                corpus_directory_path = os.path.join(tmp_directory_path, 'corpus')
                examples = toy.generate_toy_examples(corpus_directory_path,
                                                     example_class=example_class,
                                                     num_examples=num_examples)

                kwargs = {}
                if example_class is ToyExample1:
                    kwargs.update(dict(data_dirpath = corpus_directory_path))

                elif example_class is ToyExample2:

                    kwargs.update(dict(src_data_dirpath = os.path.join(corpus_directory_path, 'src'),
                                       tgt_data_dirpath = os.path.join(corpus_directory_path, 'tgt'),
                                       chunk_size_in_bins = 5))

                for i, ratios in enumerate(various_ratios):

                    save_directory_path = os.path.join(tmp_directory_path, 'tfrecords_%d' % i)

                    tf_factory.generate_and_save_train_eval_test_tfrecords_files_for_examples(save_directory_path,
                                                                                              examples,
                                                                                              ratios,
                                                                                              **kwargs)

                    # check that we have what was expected
                    expect_test_set = not(type(ratios) is float or len(ratios) == 2)
                    self._test_train_eval_test_sets_directories(save_directory_path, expect_test_set=expect_test_set)



    def test_generate_and_save_tfrecords_files_for_examples_file(self):
        """
        Here we create a corpus and save its csv file, that we use to create the train/eval/test tfrecords sets.

        This test the case where we want to create a train/eval/test sets from the original metadata.

        """

        example_classes = [ToyExample1, ToyExample2]

        for example_class in example_classes:

            with tempfile.TemporaryDirectory() as tmp_directory_path:
                #tmp_directory_path = '.'
                num_examples = 13
                ratios = [0.5, 0.3, 0.2]
                corpus_directory_path = os.path.join(tmp_directory_path, 'corpus')
                examples = toy.generate_toy_examples(corpus_directory_path,
                                                     example_class=example_class,
                                                     num_examples=num_examples)

                kwargs = {}
                if example_class is ToyExample1:
                    kwargs.update(dict(data_dirpath = corpus_directory_path))

                elif example_class is ToyExample2:

                    kwargs.update(dict(src_data_dirpath = os.path.join(corpus_directory_path, 'src'),
                                       tgt_data_dirpath = os.path.join(corpus_directory_path, 'tgt'),
                                       chunk_size_in_bins = 5))

                examples_list_filepath = os.path.join(corpus_directory_path, cts.EXAMPLES_LIST_FILENAME)
                Example.to_csv_file(examples_list_filepath, examples)

                # create train/eval/test sets.
                save_directory_path = os.path.join(tmp_directory_path, 'tfrecords')

                tf_factory.generate_and_save_train_eval_test_tfrecords_files(save_directory_path,
                                                                             example_class,
                                                                             examples_list_filepath,
                                                                             ratios,
                                                                             **kwargs)

                self._test_train_eval_test_sets_directories(save_directory_path, expect_config_file=True)


    # common tests routines

    def _test_train_eval_test_sets_directories(self, save_directory_path, expect_test_set=True, expect_config_file=False):

        # check that we have the examples.csv files. This should be the case as ToyExample implements to_csv_row.
        subdir_names = [cts.TRAIN_DIRECTORY_NAME, cts.EVAL_DIRECTORY_NAME]
        if expect_test_set:
            subdir_names.append(cts.TEST_DIRECTORY_NAME)

        for subdir_name in subdir_names:
            subdir_path = os.path.join(save_directory_path, subdir_name)
            self.assertTrue(os.path.exists(subdir_path))
            fp = tf_factory.get_examples_list_filepaths(subdir_path)
            self.assertTrue(os.path.exists(fp))

        # check that we have created a config file
        if expect_config_file:
            self.assertTrue(os.path.exists(os.path.join(save_directory_path, cts.EXAMPLES_TFRECORD_FILES_CONFIG_FILENAME)))



if __name__ == '__main__':
    unittest.main()
