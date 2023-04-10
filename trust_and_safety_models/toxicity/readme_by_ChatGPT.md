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

