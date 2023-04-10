The code in `helpers.py` provides several utility functions for machine learning model training. The functions and their descriptions are summarized in the following table:



| Function Name | Description | Parameters | Returns |
| --- | --- | --- | --- |
| upload\_model | Uploads the trained model to a Google Cloud Storage bucket and returns the path to the weights directory | `full_gcs_model_path` (str): path to the trained model in a Google Cloud Storage bucket | weights directory path (str) |
| compute\_precision\_fixed\_recall | Computes precision at a fixed recall level for a binary classification model | `labels` (array-like): true binary labels, `preds` (array-like): predicted probabilities of positive class, `fixed_recall` (float): desired recall level | precision (float), threshold (float) |
| load\_inference\_func | Loads a saved TensorFlow model for inference | `model_folder` (str): path to the saved model folder | TensorFlow inference function |
| execute\_query | Executes a query on a BigQuery client and returns the result as a Pandas DataFrame | `client` (google.cloud.bigquery.client.Client): BigQuery client, `query` (str): SQL query | Pandas DataFrame |
| execute\_command | Executes a shell command | `cmd` (str): command to execute | None |
| check\_gpu | Checks for the availability of a GPU and Tensorflow GPU installation | None | None |
| set\_seeds | Sets the random seed for NumPy, Python, and TensorFlow for reproducibility | `seed` (int): random seed | None |

The functions have various inputs and outputs as described in the table above.

