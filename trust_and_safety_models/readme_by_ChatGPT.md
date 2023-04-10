The following is the documentaion and summary for directory abusive

 Summary of abusive\_model.py
============================

The `abusive_model.py` file sets up the environment and defines the necessary components for training a machine learning model to classify abusive language in text data. The code imports various libraries, defines several variables and objects used for training a machine learning model, sets up GPU memory growth, loads training and validation data using BigQueryFeatureLoader, and prepares the data for input into a neural network model.

The code defines a prototype of the model using a set of features, including both categorical and continuous features. It then sets up various parameters for the model training process, such as the batch size, learning rate, optimizer type, and number of epochs. The labeled data is parsed and a MirroredStrategy object is created for distributing the training process across multiple GPUs.

The code sets up the neural network model architecture and compiles the model with a custom loss function and metrics. Additionally, it sets up Weights & Biases (W&B) for experiment tracking and logging.

The `abusive_model.py` file takes no parameters and returns no output. Its purpose is to define the necessary components for training a machine learning model to classify abusive language in text data.



| Function/Variable | Description |
| --- | --- |
| TensorFlow | Library for building and training machine learning models |
| BigQueryFeatureLoader | Object for loading training and validation data |
| Features | Set of features used for training the model, including both categorical and continuous features |
| MirroredStrategy | Object for distributing the training process across multiple GPUs |
| Batch size, learning rate, optimizer type, and number of epochs | Parameters for the model training process |
| Loss function and metrics | Custom loss function and metrics used for model compilation |
| Weights & Biases (W&B) | Experiment tracking and logging tool |

 


The following is the documentaion and summary for directory nsfw

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

 


The following is the documentaion and summary for directory toxicity

 Summary of `text_processing` directory
--------------------------------------



| Filename | Description |
| --- | --- |
| `clean_text.py` | Defines `TextCleaner` class to clean text data by removing regular expressions and non-ASCII characters. |
| `embedding.py` | Defines `TextEmbedder` class to embed text data using a tokenizer and embedding model. |
| `tokenization.py` | Defines `Tokenizer` class to tokenize text data by splitting with regular expressions and optionally lowercasing. |
| `word2vec.py` | Defines `Word2Vec` class to train a word2vec model and generate word embeddings for tokenized sentences. |

Summary of `train.py` file
--------------------------

The `train.py` file defines a `Trainer` class that encapsulates all the parameters, data, and logic necessary to train a machine learning model on toxicity data. It initializes with various hyperparameters and loads relevant modules, sets up logging and checkpoint paths, initializes mini-batch loaders, and sets up learning rate schedules.

The `get_callbacks()` method of the `Trainer` class returns a list of callback functions used by `model.fit()` to monitor training progress and update the model at the end of each epoch.

The `_init_dirnames()` method of the `Trainer` class sets up logging and checkpoint paths based on various hyperparameters and experiment IDs. It also creates a unique experiment name based on the current timestamp, the language of the data, the optimizer name, the learning rate, the weight decay, the mini-batch size, the percentage of toxic comments in the training data, the number of training epochs, and the random seed.

The `check_gpu()` function is called to check if a GPU is available for training the model.

### `Trainer` class



| Method | Description |
| --- | --- |
| `__init__(self, train_data, val_data, test_data, model_class, tokenizer, word_embedding, vocab_size, emb_size, mb_size, train_epochs, optimizer_name, weight_decay, learning_rate, lr_decay, lr_step_size, patience, random_seed)` | Initializes the `Trainer` class with specified hyperparameters. |
| `get_callbacks(self)` | Returns a list of callback functions used by `model.fit()` to monitor training progress and update the model at the end of each epoch. |
| `_init_dirnames(self)` | Sets up logging and checkpoint paths based on various hyperparameters and experiment IDs, and creates a unique experiment name. |
| `train(self)` | Trains the machine learning model on the specified data using the specified hyperparameters, and returns the trained model. |

### `check_gpu()` function



| Parameter | Description |
| --- | --- |
| None | Checks if a GPU is available for training the model. |
| Returns | None |

 

