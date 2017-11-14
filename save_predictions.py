import numpy as np
import pandas as pd

def save_predictions(dataframe, predictions, columns_for_predictions, filename):
    dataframe_to_save = dataframe.copy()
    for column_id, column in enumerate(columns_for_predictions):
    	dataframe_to_save.loc[:,column] = predictions[:,column_id]
    dataframe_to_save.to_csv(filename, index=False)
    return 1
