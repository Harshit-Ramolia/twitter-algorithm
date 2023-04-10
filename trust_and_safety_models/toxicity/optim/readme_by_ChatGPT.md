Summary of Python Scripts
-------------------------



| Script Name | Description |
| --- | --- |
| callbacks.py | Defines several callback classes for use in TensorFlow/Keras models. `NothingCallback` prints information about the training process during training. `ControlledStoppingCheckpointCallback` and `SyncingTensorBoard` perform additional actions after each epoch. `GradientLoggingTensorBoard` logs the gradient norm of the model's trainable weights every `freq` batches. `AdditionalResultLogger` computes various evaluation metrics on the model's predictions. |
| losses.py | Defines two custom loss functions for use in TensorFlow/Keras models. `InvKLD` calculates the inverse Kullback-Leibler divergence between the true and predicted probability distributions of the targets. `MaskedBCE` calculates the binary cross-entropy loss between the true and predicted targets, but only on the subset of the targets that are not equal to -1. |
| schedulers.py | Defines a WarmUp learning rate schedule for use with TensorFlow/Keras optimizers. The schedule gradually increases the learning rate during a warm-up period before decaying according to a user-specified decay function. |

### `callbacks.py`

#### `NothingCallback`

* Class that inherits from `tf.keras.callbacks.Callback`
* Overrides `on_epoch_begin`, `on_epoch_end`, and `on_train_batch_end` to print information about the training process during training.

#### `ControlledStoppingCheckpointCallback`

* Class that inherits from `tf.keras.callbacks.ModelCheckpoint`
* Overrides `on_epoch_end` to perform additional actions after each epoch.

#### `SyncingTensorBoard`

* Class that inherits from `tf.keras.callbacks.TensorBoard`
* Overrides `on_epoch_end` to perform additional actions after each epoch.
* Has additional method `synchronize` which performs additional actions.

#### `GradientLoggingTensorBoard`

* Class that inherits from `tf.keras.callbacks.TensorBoard`
* Overrides `on_train_batch_end` to log the gradient norm of the model's trainable weights every `freq` batches.
* Has additional method `_log_gradients` which performs additional actions.

#### `AdditionalResultLogger`

* Class that inherits from `tf.keras.callbacks.Callback`
* Initialized with various arguments, including `data`, `set_`, `fixed_recall`, `from_logits`, `dataset_transform_func`, `batch_size`, and `dual_head`.
* Defines a method called `additional_evaluations` which performs various evaluations on the model's predictions.
* Overrides `on_epoch_end` and `on_train_batch_end` to call `additional_evaluations` after each epoch and every 2000 batches, respectively.
* Uses `labels` and `data` attributes to compute the various evaluation metrics.

### `losses.py`

#### `InvKLD`

* Custom loss function for use in TensorFlow/Keras models.
* Calculates the inverse Kullback-Leibler divergence between the true and predicted probability distributions of the targets.
* Takes in `y_true` and `y_pred` as input and returns the inverse KL divergence as a scalar tensor.

#### `MaskedBCE`

* Custom loss function for use in TensorFlow/Keras models.
* Calculates the binary cross-entropy loss between the true and predicted targets, but only on the subset of the targets that are not equal to -1.
* Takes in `y_true` and `y_pred` as input and returns the masked binary cross-entropy loss as a scalar tensor.

### `schedulers.py`

#### `WarmUp`

* Subclass of `tf.keras.optimizers.schedules.LearningRateSchedule`.
* Takes in `initial_learning_rate`, `decay_schedule_fn`, `warmup_steps`, `power`, and an optional `name`.
* Computes the learning rate for the current training step based on whether it falls within the warm-up period
