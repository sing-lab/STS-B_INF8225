"""
Import train and set data
"""

import pandas as pd
import os
from sklearn.model_selection import train_test_split

def loader_train():
    filepath = os.path.join(os.getcwd(), 'input', 'inf8460-sts-bA19', 'sts-b_train.csv') # locate train data
    
    df = pd.read_csv(filepath) # Read file
    df.index = df['id'] # Change index name
    df = df.drop(['id'], axis=1) # Remove 'id' column, unuseful because already in index
    
    dfX, y = df[['sentence1', 'sentence2']], df['score'] # Split features and targets
    return dfX, y

def loader_test():
    filepath = os.path.join(os.getcwd(), 'input', 'inf8460-sts-bA19', 'sts-b_test.csv') # locate train data
    
    df = pd.read_csv(filepath)
    df.index = df['id']
    
    dfX = df[['sentence1', 'sentence2']] # Select features
    return dfX

def train_validation_split(dfX, y, validation_size=0.25):
    """Does exactly the same job as train_test_split. Just to be clearer."""
    return train_test_split(dfX, y, test_size=validation_size)
