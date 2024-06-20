import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import yaml
import logging


logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

formatter = logging.Formatter()

def load_params(params_path: str) -> float:
    try:
        with open(params_path,'r') as file:
            params = yaml.safe_load(file)
        test_size = params['data_ingestion']['test_size']
        return test_size
    except FileNotFoundError:
        print(f"Error: The File {params_path} Was not found")
        raise
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse the yaml File {params_path}")
        print(e)
        raise
    except Exception as e:
        print(f"Unexcepted Error occured: ")
        print(e)
        raise


def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except FileNotFoundError:
        print(f"error: the file {url} was not found")
        raise
    except Exception as e:
        print(f"error: {e}")
        raise
def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns=['tweet_id'],inplace=True)
        final_df = df[df['sentiment'].isin(['neutral','sadness'])]

        final_df['sentiment'] = final_df['sentiment'].map({"neutral":1,"sadness":0})

        return final_df
    
    except TypeError:
        print("df is not of correct type to perform these operations")

    except AttributeError:
        print("sentiment is not present in DataFrame")

    except Exception as e:
        print("Error: {e}")


def save_data(data_path: str,train_data: pd.DataFrame,test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path)

        train_data.to_csv(os.path.join(data_path,"train.csv"))
        test_data.to_csv(os.path.join(data_path,"test.csv"))
    except FileExistsError:
        train_data.to_csv(os.path.join(data_path,"train.csv"))
        test_data.to_csv(os.path.join(data_path,"test.csv"))
    
    except Exception as e:
        print(f"Error: {e}")

def main() -> None:
    test_size = load_params('params.yaml')
    df = read_data(r'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
    final_df = process_data(df)
    train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)
    data_path = os.path.join("data","raw")
    save_data(data_path,train_data,test_data)




if __name__ == "__main__":
    main()

