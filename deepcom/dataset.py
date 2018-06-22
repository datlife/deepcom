import tensorflow as tf

def data_genenerator(X, Y, batch_size, shuffle=True):
    """A Tensorflow way to load data for training.

    Args:
        X - ndarray - inputs 
        Y - ndarray - ground truths
        batch_size - int - number of inputs to load 
          into model per run.
        shuffle - a boolean - to shuffle data or not
           should be off during Evaluation/Testing.
    
    Return
       dataset - a tf.data.Dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.prefetch(batch_size*3)
    if shuffle:
        dataset = dataset.shuffle(1000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
    return dataset


