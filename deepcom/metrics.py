"""Contains metrics to monitor during training/evaluation"""
import tensorflow as tf

def BER(y, y_pred):
  """Measure Bit Error Rate (BER)
  Args:
    y - tf.Tensor - shape (batch_size, K, 1)
    y_pred - tf.Tensor - shape (batch_size, K, 1)

  Returns:
    ber - a tf.float - represents bit error rate
        in a batch.
  """
  hamming_distances =  tf.cast(tf.not_equal(y, tf.round(y_pred)), tf.int32)
  ber = tf.reduce_sum(hamming_distances) / tf.size(y)
  return ber

def BLER(y, y_pred):
    """Measure Bit Block Error Rate (BER)
    Args:
      y - tf.Tensor - shape (batch_size, K, 1)
      y_pred - tf.Tensor - shape (batch_size, K, 1)

    Returns:
      bler - a tf.float - represents bit block error rate
          in a batch.
    """
    num_blocks_per_batch = tf.cast(tf.shape(y)[0], tf.int64)
    hamming_distances =  tf.cast(tf.not_equal(y, tf.round(y_pred)), tf.int32)
    bler = tf.count_nonzero(tf.reduce_sum(hamming_distances, axis=1)) / num_blocks_per_batch
    return bler

class TrainValTensorBoard(tf.keras.callbacks.TensorBoard):
    """Custom Tensorboard Callback to plot training and testing into one graph.
    https://stackoverflow.com/questions/47877475/keras-tensorboard-plot-train-and-validation-scalars-in-a-same-figure
    """
    def __init__(self, log_dir='./logs', **kwargs):
        import os
        # Make the original `TensorBoard` log to a subdirectory 'training'
        training_log_dir = os.path.join(log_dir, 'training')
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

        # Log the validation metrics to a separate subdirectory
        self.val_log_dir = os.path.join(log_dir, 'validation')

    def set_model(self, model):
        # Setup writer for validation metrics
        self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def on_epoch_end(self, epoch, logs=None):
        # Pop the validation logs and handle them separately with
        # `self.val_writer`. Also rename the keys so that they can
        # be plotted on the same figure with the training metrics
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if k.startswith('val_')}
        for name, value in val_logs.items():
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value.item()
            summary_value.tag = name
            self.val_writer.add_summary(summary, epoch)
        self.val_writer.flush()

        # Pass the remaining logs to `TensorBoard.on_epoch_end`
        logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
        super(TrainValTensorBoard, self).on_epoch_end(epoch, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()