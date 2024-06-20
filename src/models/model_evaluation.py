import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_auc_score



def read_model(model_path,test_data_path):
    try:
        model = pickle.load(open(model_path,'rb'))
        test_data = pd.read_csv(test_data_path)   # 
        return model,test_data
    except Exception as e:
        print("Some error occuered ",e)

def perform(model,test_data):
    try:
        X_test = test_data.iloc[:,0:-1].values
        y_test = test_data.iloc[:,-1].values

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:,1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        return accuracy,precision,recall,auc
    except Exception as e:
        print(f"Some error occured: {e}")

def get_metric(accuracy,precision,recall,auc):
    try:
        metrics_dict={
            'accuracy':accuracy,
            'precision':precision,
            'recall':recall,
            'auc2':auc
        }

        return metrics_dict
    except Exception as e:
        print(f"Some error occured: {e}")


def save_metric(metric):
    try:
        with open('reports/metrics.json', 'w') as file:
            json.dump(metric, file, indent=4)

    except Exception as e:
        print(f"Some error occured: {e}")



def main():
    model,test_data = read_model('models/model.pkl',"./data/features/test_bow.csv")
    accuracy,precision,recall,auc = perform(model,test_data)
    metrics_dict = get_metric(accuracy,precision,recall,auc)
    save_metric(metrics_dict)


if __name__ == "__main__":
    main()