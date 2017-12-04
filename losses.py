import numpy as np

def triplet_loss(y_true, y_pred):
    # y_true: 2 x n_samples x n_classes(=1)
    margin = 0.05
    y_pred_human = y_pred[0]
    y_pred_bot = y_pred[1]
    n_samples = y_pred_human.shape[0]
    # margin = 0.05
    # result = np.amax(np.vstack(( y_pred_human - y_pred_bot + margin, np.zeros((n_samples,1)) )), axis=1)
    result = y_pred_bot - y_pred_human - margin
    return result
