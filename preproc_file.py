import numpy as np
import pandas as pd

# train = pd.read_csv("/home/dilyara/Documents/GitHub/deeppavlov/build/insults/train.csv")
# valid = pd.read_csv("/home/dilyara/Documents/GitHub/deeppavlov/build/insults/test_with_solutions.csv")
# test = pd.read_csv("/home/dilyara/Documents/GitHub/deeppavlov/build/insults/impermium_verification_labels.csv")
#
# print(train.head())
# print(valid.head())
# print(test.head())
#
# train["Usage"] = "Train"
#
# train_valid = train.append(valid)
#
# train_valid["id"] = np.arange(train_valid.shape[0])
# print("Train + valid:\n\n",train_valid.head())
# train_valid = train_valid.loc[:,['id', 'Insult', 'Date', 'Comment', 'Usage']]
# train_valid.to_csv("/home/dilyara/Documents/GitHub/deeppavlov/build/insults/full_train_as_test.csv", index=False)

train = pd.read_csv("/home/dilyara/Documents/GitHub/deeppavlov/build/insults/train.csv")
valid = pd.read_csv("/home/dilyara/Documents/GitHub/deeppavlov/build/insults/test_with_solutions.csv")
test = pd.read_csv("/home/dilyara/Documents/GitHub/deeppavlov/build/insults/impermium_verification_labels.csv")

print(train.head())
print(valid.head())
print(test.head())

train["Usage"] = "Train"

train_valid = train.append(valid)

train_valid["id"] = np.arange(train_valid.shape[0])
print("Train + valid:\n\n",train_valid.head())
train_valid = train_valid.loc[:,['id', 'Insult', 'Date', 'Comment', 'Usage']]
train_valid.to_csv("/home/dilyara/Documents/GitHub/deeppavlov/build/insults/full_train_as_test.csv", index=False)



