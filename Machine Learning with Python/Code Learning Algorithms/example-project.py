#---README---

# I decided to have a pure Python file in this folder for the example Titanic Project
# So that you can look at the code all in one place
# I would highly recommend using the Jupyter Notebook, as it allows for better useage, and I've got example outputs embedded in it


#I'll put in a few notes in here as well.

#---Start of Code---

#Importing the required libraries;

import numpy as np #For manipulation of datasets, and general tools
import pandas as pd #For neater and more efficient dataset usage
import matplotlib.pyplot as plt #for plotting
from IPython.display import clear_output #For some better and clearer output
from six.moves import urllib #For potential backwards compatibility with certain modules

#Importing TenserFlow

import tensorflow.compat.v2.feature_column as fc

import tensorflow as tf

# Loading the dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')

y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')



#---Machine Learning Model---


#Seperating columns, and creating 'Feature' Columns:
CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))