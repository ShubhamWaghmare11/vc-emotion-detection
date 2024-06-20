import numpy as np
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import GradientBoostingClassifier

def params_load(path):
    try:    
        params = yaml.safe_load(open(path,'r'))['model_building']
        return params['n_estimators'],params['learning_rate']
    except Exception as e:
        print(f"Some error occured: {e}")



def read_data(path):
    try:
        train_data = pd.read_csv(path)
        return train_data
    except Exception as e:
        print(f"Some error occured: {e}")


def split_data(train_data):
    try:
        X_train = train_data.iloc[:,0:-1].values
        y_train = train_data.iloc[:,-1].values
        return X_train,y_train
    
    except Exception as e:
        print(f"Some error occured: {e}")


def model_training(X_train,y_train,n_estimators,learning_rate):
    try:
        clf = GradientBoostingClassifier(n_estimators=n_estimators,learning_rate=learning_rate)
        clf.fit(X_train, y_train)
        return clf
    
    except Exception as e:
        print(f"Some error occured: {e}")
        raise


def dump_model(model):
    try:
        pickle.dump(model, open('model.pkl','wb'))
    
    except Exception as e:
        print(f"Some error occured: {e}")


def main():
    n_estimator,learning_rate = params_load("params.yaml")
    train_data = read_data('./data/features/train_bow.csv')
    x_train,y_train = split_data(train_data)
    model = model_training(x_train,y_train,n_estimator,learning_rate)
    dump_model(model)


if __name__ == "__main__":
    main()