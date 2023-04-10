Summary of nsfw\_media.py
-------------------------

The nsfw\_media.py script is a machine learning pipeline that reads in TFRecord files of training, test, and validation data for image classification, preprocesses the data, and creates TensorFlow datasets. It also applies resampling if `do_resample` is set to `True`. After creating the datasets, the script defines metrics to evaluate the models, including `tf.keras.metrics.PrecisionAtRecall` and `tf.keras.metrics.AUC`.

### Parameters

* `train_tfrecords`: Path to the TFRecord file containing the training data
* `val_tfrecords`: Path to the TFRecord file containing the validation data
* `test_tfrecords`: Path to the TFRecord file containing the test data
* `batch_size`: Batch size used for training and evaluation
* `img_size`: Target image size for preprocessing
* `do_resample`: Whether or not to apply resampling

### Returns

The script does not return anything, but it creates and trains a TensorFlow model for image classification.

Summary of nsfw\_text.py
------------------------

The nsfw\_text.py script is an implementation of a model that classifies tweets into two categories: safe for work (not NSFW) and not safe for work (NSFW).

The script imports necessary libraries, including TensorFlow and pandas. It then defines several variables and functions, including regular expressions used to clean the tweets and a function that applies the model to predict the NSFW probability of a tweet. The script loads and cleans the tweet text, then splits it into training and validation sets. The training set is converted into a `tf.data.Dataset` object using `df_to_ds`.

The model is defined using the `TextEncoder` class, which uses a pre-trained language model to encode the tweets into fixed-length vectors. The model is trained using `model.fit`, and the training history is stored in `history`. Finally, the script applies the model to the validation set and generates a classification report and a precision-recall curve.

### Parameters

* `data_path`: Path to the CSV file containing the tweet data
* `max_length`: Maximum length of the tweet text
* `batch_size`: Batch size used for training and evaluation
* `epochs`: Number of epochs to train the model

### Returns

The script does not return anything, but it trains and evaluates a TensorFlow model for tweet classification, and generates a classification report and a precision-recall curve.

