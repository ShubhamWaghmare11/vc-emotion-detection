import numpy as np
import pandas as pd

import os

from sklearn.feature_extraction.text import CountVectorizer

import yaml



def params_load(path):
    try:
        max_features = yaml.safe_load(open(path,'r'))['feature_engineering']['max_features']
        return max_features
    
    except FileNotFoundError:
        print(f"Error: the file {path} was not found")

    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the yaml file {path}")

    except Exception as e:
        print(f"unexpected error occured: {e}")





def read_clean_data(path):
    try:
        # fetch the data from data/processed
        train_data = pd.read_csv(os.path.join(path,"train_processed.csv"))
        test_data = pd.read_csv(os.path.join(path,"test_processed.csv"))

        train_data.fillna('',inplace=True)
        test_data.fillna('',inplace=True)

        return train_data,test_data
    
    except Exception as e:
        print("Some error occured: {e}")

def train_test_split(train_data,test_data):
    try:
        X_train = train_data['content'].values
        y_train = train_data['sentiment'].values

        X_test = test_data['content'].values
        y_test = test_data['sentiment'].values
        return X_train,X_test,y_train,y_test
    
    except Exception as e:
        print(f"Some error occured: {e}")


def vectorization(X_train,X_test,y_train,y_test,max_features):
    try:
        # Apply Bag of Words (CountVectorizer)
        vectorizer = CountVectorizer(max_features=max_features)

        # Fit the vectorizer on the training data and transform it
        X_train_bow = vectorizer.fit_transform(X_train)

        # Transform the test data using the same vectorizer
        X_test_bow = vectorizer.transform(X_test)

        train_df = pd.DataFrame(X_train_bow.toarray())

        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())

        test_df['label'] = y_test

        return train_df,test_df
    
    except Exception as e:
        print(f"Some error occured: {e}")



def save_data(path,train_data,test_data):
    try:
        os.makedirs(path)

        train_data.to_csv(os.path.join(path,"train_bow.csv"))
        test_data.to_csv(os.path.join(path,"test_bow.csv"))

    except Exception as e:
        print(f"Error: Some Error occured: {e}")


def main():
    max_features = params_load("params.yaml")
    train_data,test_data =  read_clean_data("data/processed")
    x_train,x_test,y_train,y_test = train_test_split(train_data,test_data)
    train_data,test_data = vectorization(x_train,x_test,y_train,y_test,max_features=max_features)
    path = os.path.join("data","features")
    save_data(path,train_data,test_data)



if __name__ == "__main__":
    main()
