Summary of Machine Learning Pipeline Codebase
---------------------------------------------

This codebase appears to be a machine learning pipeline for training a model to predict the toxicity of tweets. The pipeline includes several modules, each with its own classes and methods for data loading, preprocessing, and training.

### `dataframe_loader.py`

The `dataframe_loader.py` module includes two classes for loading data from BigQuery and performing sampling on the loaded data:



| Class | Description |
| --- | --- |
| `ENLoader` | Base class for loading data from BigQuery. Includes methods for producing a SQL query and loading data from either BigQuery or a pickle file. |
| `ENLoaderWithSampling` | Derived class of `ENLoader` that includes additional methods for performing data sampling. |

### `data_preprocessing.py`

The `data_preprocessing.py` module defines several classes for cleaning and preprocessing dataframes of text data:



| Class | Description |
| --- | --- |
| `DataframeCleaner` | Abstract base class for cleaning and preprocessing dataframes. Defines methods for cleaning, systematic preprocessing, and postprocessing. |
| `DefaultENNoPreprocessor` | Subclass of `DataframeCleaner` for English language dataframes. Implements a default cleaning and preprocessing pipeline, including removing duplicates and filtering out tweets with media. |
| `DefaultENPreprocessor` | Subclass of `DefaultENNoPreprocessor` that includes additional cleaning steps, such as replacing URLs and mentions with placeholders. |
| `Defaulti18nPreprocessor` | Subclass of `DataframeCleaner` for non-English language dataframes. Implements a default cleaning pipeline that removes URLs, mentions, and newlines. |

### `mb_generator.py`

The `mb_generator.py` module includes a class for creating a balanced mini-batch loader for a toxicity classification task:



| Class | Description |
| --- | --- |
| `BalancedMiniBatchLoader` | Takes in several parameters, including `fold`, `mb_size`, `perc_training_tox`, `seed`, and `scope` and `project`. Methods include `_load_tokenizer`, `get_outer_fold`, `make_huggingface_tensorflow_ds`, and `make_pure_tensorflow_ds`. |

Overall, this codebase appears to be a comprehensive machine learning pipeline for training a model to predict the toxicity of tweets, with modules for data loading, preprocessing, and training.

