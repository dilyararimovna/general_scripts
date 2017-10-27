import numpy as np
import pandas as pd

def save_predictions(dataframe, predictions, columns_for_preds, filename):
    dataframe_to_save = dataframe.copy()
    dataframe_to_save.loc[:,columns_for_predictions] = predictions
    dataframe_to_save.to_csv(filename, index=False)
    return 1
