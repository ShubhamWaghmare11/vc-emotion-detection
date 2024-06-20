import numpy as np
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords
import os
from nltk.stem import SnowballStemmer,WordNetLemmatizer


def read_data(train_url,test_url):
    train_df = pd.read_csv(train_url)
    test_df = pd.read_csv(test_url)
    return train_df,test_df


nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):
    stop_words = set(stopwords.words("english"))
    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(df):
    df.content=df.content.apply(lambda content : lower_case(content))
    df.content=df.content.apply(lambda content : remove_stop_words(content))
    df.content=df.content.apply(lambda content : removing_numbers(content))
    df.content=df.content.apply(lambda content : removing_punctuations(content))
    df.content=df.content.apply(lambda content : removing_urls(content))
    df.content=df.content.apply(lambda content : lemmatization(content))
    return df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= removing_numbers(sentence)
    sentence= removing_punctuations(sentence)
    sentence= removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence





def save_data(data_path,train_data,test_data):

    try:
        os.makedirs(data_path)

        train_data.to_csv(os.path.join(data_path,"train_processed.csv"))
        test_data.to_csv(os.path.join(data_path,"test_processed.csv"))

    except FileExistsError:
        train_data.to_csv(os.path.join(data_path,"train_processed.csv"))
        test_data.to_csv(os.path.join(data_path,"test_processed.csv"))

    except Exception as e:
        print(f"Error: {e}")
        raise


def main():
    train_data,test_data = read_data(r"data/raw/train.csv",r"data/raw/test.csv")
    train_processed_data = normalize_text(train_data)
    test_processed_data = normalize_text(test_data)
    save_data(r"data/processed",train_processed_data,test_processed_data)




if __name__ == "__main__":
    main()