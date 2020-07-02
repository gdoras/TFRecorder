# TFRecorder


The goal of this project is to ease the creation of [tfrecord](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfrecord_files_in_python) files for
 training, evaluation (and optionally testing) a Tensorflow model, hiding
 boilerplate code under the hood.

## Usage

[Subclassing `Example`](#subclassing-example)  
* [Using the `@tfrecordable` decorator](#using-tfrecordable-decorator)   
* [Overriding the `load` method](#overriding-the-load-method)  
* [Overriding the `split` method (optionally)](#overriding-the-split-method)  

[Instantiating Example subclass](#instantiating-example-subclass)
* [Overriding the `from_csv_row` method](#overriding-the-from_csv_row-method)
* [Overriding the `to_csv_row` method](#overriding-the-to_csv_row-method)
* [Preparing the `examples.csv` file](#preparing-the-examplescsv-file)

[Generating the tfrecord files](#generating-the-tfrecord-files)

[Generating a tf.data.Dataset](#generating-a-tfdatadataset)

The `Example` class shall be subclassed to represent the data of the 
task at hand. Each instance of your `Example` subclass will then be 
processed by the `factory` and converted into one (or various) 
[tf.Example](https://www.tensorflow.org/tutorials/load_data/tfrecord#tfexample) 
instance(s) and stored in tfrecord files.

### Subclassing `Example`

Using `tfrecords` files probably means that your data can not fit into 
memory. Your subclass should therefore be instantiated only with its 
lightweight attributes, and defer loading of its memory consuming 
attributes for a later stage (see the `load` method below).

For instance, consider a classification task where a float array is used as source
data and an integer is used as a target label:

```python
import numpy as np
import os

from tfrecorder.helpers.decorator import tfrecordable
from tfrecorder.helpers.marshaller import Example

class ToyExample(Example):

    def __init__(self, name, label):

        super(ToyExample, self).__init__()
        self.name = name
        self._label = label # this is an int, so we assign it when instantiating
        self._data = None # this is a float array, so we defer loading
```

#### Using `@tfrecordable` decorator 

The `@tfrecordable` decorator indicates which attributes of your `Example` 
subclass shall be serialized into tfrecords files. It is used in the same fashion as the built-in
`@property` decorator, but is passed the type of the corresponding 
serializable attribute.

For our classification task, only the `_label` and `_data` attributes will
be serialized into `tfrecords` files, but not the `name` attribute: 

```python
    # CAUTION: the attributes must be declared in the same order than expected when parsing in tf.Dataset
    @tfrecordable(dtype=Example.Field.TYPE_INT32)
    def label(self):
        return self._label
    
    @label.setter
    def label(self, val):
        self._label = val
    
    @tfrecordable(dtype=Example.Field.TYPE_ARRAY_FLOAT32)
    def data(self):
        return self._data
        
    @data.setter
    def data(self, val):
        self._data = val
```

The `dtype` parameter of the `@tfrecordable` decorator is used to tell Tensorflow
which data type to use when serializing and de-serializing. The supported types 
are:

<table>
<tr>
    <td>TYPE_BOOL</td>
    <td>TYPE_ARRAY_INT32</td>
    <td>TYPE_INT32</td>
    <td>TYPE_FLOAT</td>
    <td>TYPE_STRING</td>
</tr>
<tr>
    <td></td>
    <td>TYPE_ARRAY_FLOAT32</td>
    <td>TYPE_INT64</td>
    <td>TYPE_DOUBLE</td>
    <td></td>
</tr>
</table>

You should use `TYPE_ARRAY_INT32` or `TYPE_ARRAY_FLOAT32` when serializing a numpy array. Note that if will
be stored as `tf.int32` or `tf.float32` in the tfrecord file.

#### Overriding the `load` method

If your `Example` subclass has some deferred loading of data, it must override
 the `load` instance method.
 
 This method will be called just before your example instance is serialized 
 into a `tfrecord` file, and should implement the logic needed to load your
 data into memory, and optionally pre-process it if needed. Pass it 
 all parameters required to load and optionally pre-process your data.
 
Finally, make sure that your data is assigned to its corresponding
 `@tfrecordable` attribute.
 
Once saved to the file, all memory used during serialization will be released. 
  
For our classification task, we could have:

```python
    def load(self, src_data_dirpath=None, normalize=False, **kwargs):
        """
        Loads the data in memory, and possibly pre-process it.
        """
        data = np.load(os.path.join(src_data_dirpath, '%s.npy' % self.name))
            
        if normalize:
            data /= np.max(data)

        self.data = data
```

#### Overriding the `split` method 

If the data of your `Example` subclass shall be split into chunks before 
serialization, it must override the `split` instance method, and wrap each
chunk into an instance of your `Example` subclass. Otherwise, don't override the `split`
instance method.

For our classification task, we could write:
```python
    def split(self, chunk_size_in_bins=5, **kwargs):
        """
	    Splits this example data into chunks, and create one ToyExample object per chunk.

	    Returns:
		    examples: list, of ToyExample objects.

	    """
        src_data = self.src_data

        chunked_examples = []
        for i in range(0, src_data.shape[0], chunk_size_in_bins):
            chunked_example = ToyExample('%s_%d' % (self.name, self._label))
            chunked_example.src_data = src_data[i:i+chunk_size_in_bins]
            chunked_examples.append(chunked_example)

        return chunked_examples
```

### Instantiating `Example` subclass

At this point, you can instantiate your subclass for your data and pass it to
one of the `factory`'s methods. It is however simpler to let TFRecorder handle
it for you. 

#### Overriding the `from_csv_row` method

Your `Example` subclass should override `from_csv_row` static method to return
a new instance of your subclass. The `row` argument will be passed by a 
csv reader, and is thus a list of strings.

```python
    @staticmethod
    def from_csv_row(row, **kwargs):
        """
        This is used to read from a csv file a single row that should be enough to recreate this Example.
        Don't forget to cast, as the csv reader returns strings only.

        Args:
            row: list, of strings, as returned by a csv reader for this row of a csv file.
            **kwargs: dict, other parameters passed by the factory. Optional.

        Returns:
            example: an instance of the Example object
        """
        return ToyExample(name=row[0], label=int(row[1]))
```

#### Overriding the `to_csv_row` method
Similarly, your `Example` subclass should override `to_csv_row`: it will
be used when saving metadata describing what is stored in the train, eval 
(and optionally test) tfrecord files.

```python
    def to_csv_row(self):
        """
        This is used to write in a csv file a single row that should be enough to recreate this Example afterward.
        """
        return [self.name, self._label]
```

#### Preparing the `examples.csv` file

The information required to instantiate your `Example` subclass shall 
be stored in a csv file that will be parsed by the `factory`. Each row
will be passed to the `from_csv_row` static method of your `Example` 
subclass.


### Generating the tfrecord files

That's it. The factory will generate the tfrecord files passing it 
your `Example` subclass, the path to your csv file and the other parameters
that are used by your subclass' `load` and maybe `split` methods.


```python
tf_factory.generate_and_save_train_eval_test_tfrecords_files(save_directory_path='/my/path/where/to/save',
                                                             example_class=ToyExample,
                                                             examples_list_filepath='/my/path/to/csv_file',
                                                             src_data_dirpath='/my/path/to/data', 
                                                             normalize=True,
                                                             chunk_size_in_bins=5)
```

The factory will save in your location in a `train`, `eval` and optionally
`test` directories the corresponding tfrecord files.

### Generating a tf.data.Dataset

The whole point of using tfrecord files is to stream them into a 
[Dataset](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)
object.

To stream the content of tfrecord files stored in a directory, create a
Dataset object with the `factory`. Each `@tfrecordable`attribute of your
 `Example` subclass will be streamed as a tuple where they appear in the 
 same order they have been declared.
 
```python
tfrecord_filepaths = tf_factory.get_tfrecord_filepaths(dirpath='my/path/where/to/save/train')

dataset = tf_factory.generate_dataset(tfrecord_filepaths)

for l in dataset:
    print(l) # label, data


```



