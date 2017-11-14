import fasttext
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support

path_to_true = '/home/dilyara/Documents/GitHub/deeppavlov/src/parlai/data/insults'
path_to_pred = '/home/dilyara/Documents/GitHub/fastText'

classifier = fasttext.supervised(path_to_true + '/train_fasttext_cls.txt', 'cls_model')

test = pd.read_csv(path_to_true + '/test.csv')
test_pred = classifier.predict_proba(test.loc[:,'Comment'].values)
test_pred = [(test_pred[i][0][0] == '1') * test_pred[i][0][1] + (test_pred[i][0][0] == '0') * (1. - test_pred[i][0][1])
             for i in range(test.shape[0])]

print('Test AUC-ROC', roc_auc_score(test.loc[:,'Insult'].values, test_pred))
print('Test Accuracy', accuracy_score(test.loc[:,'Insult'].values, np.round(test_pred)))
print('Test f1', f1_score(test.loc[:,'Insult'].values, np.round(test_pred)))
print('Test precision recall fscore support',
      precision_recall_fscore_support(test.loc[:,'Insult'].values, np.round(test_pred), average='binary'))

train = pd.read_csv(path_to_true + '/train.csv')
train_pred = classifier.predict_proba(train.loc[:,'Comment'].values)
train_pred = [(train_pred[i][0][0] == '1') * train_pred[i][0][1] + (train_pred[i][0][0] == '0') * (1. - train_pred[i][0][1])
              for i in range(train.shape[0])]
print('Train AUC-ROC', roc_auc_score(train.loc[:,'Insult'].values, train_pred))
print('Train Accuracy', accuracy_score(train.loc[:,'Insult'].values, np.round(train_pred)))
print('Train f1', f1_score(train.loc[:,'Insult'].values, np.round(train_pred)))
print('Train precision recall fscore support',
      precision_recall_fscore_support(train.loc[:,'Insult'].values, np.round(train_pred), average='binary'))
