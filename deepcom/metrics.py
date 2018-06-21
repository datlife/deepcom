import numpy as np

# =============================
# Define Metrics
# =============================
def compute_BER(y, y_pred):
    """Measure Bit Error Rate (BER)"""
    assert y.shape == y_pred.shape, \
    "Prediction and Ground truth must have same shape.\n"\
    "Expected:{0} Actual:{1}".format(y.shape, y_pred.shape)

    error = np.where(y != y_pred)
    error_sum = float(len(error[0]))
    ber = error_sum / np.product(y.shape)
    return ber

def compute_accuracy(y, y_pred):

    # Make sure shapes are matched
    assert y.shape == y_pred.shape, \
    "Prediction and Ground truth must have same shape.\n"\
    "Expected:{0} Actual:{1}".format(y.shape, y_pred.shape)

    return np.sum(y == y_pred) / np.product(y.shape)
