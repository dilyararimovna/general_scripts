import numpy as np
import pandas as pd


train = pd.read_csv("./ParlAI/data/insults/train.csv")
test = pd.read_csv("./ParlAI/data/insults/test.csv")

count = 0
for i in range(test.shape[0]):
    for j in range(train.shape[0]):
        if test.loc[i,'Comment'] == train.loc[j,'Comment']:
            count += 1
            print('Similarity!', i)

print('Number of similarity:',count)
