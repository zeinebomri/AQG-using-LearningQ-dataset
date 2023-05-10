import subprocess

def install(name):
    subprocess.call(['pip', 'install', name])

install ("nltk")
install ("contractions")

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import contractions
import pandas as pd
import re

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

train = pd.read_csv("train.csv", usecols = ['context','question'],engine='python',error_bad_lines=False)
valid = pd.read_csv("valid.csv", usecols = ['context','question'],engine='python',error_bad_lines=False)
test = pd.read_csv("test.csv", usecols = ['context','question'],engine='python',error_bad_lines=False)

def remove_non_ascii(text):
    return re.sub(r'[^\x00-\x7f]',r'', text)

def remove_URL(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_html(text):
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

def preprocess(data):
    data = data.replace(" '","'") #remove extra spaces
    data=contractions.fix(data) #example: let's -->let us
    data=remove_non_ascii(data) 
    data=remove_URL(data) 
    data=remove_html(data)
    data=data.lower()
    return data

train=train.dropna()
valid=valid.dropna()
test=test.dropna()

#remove duplicates if any
train = train.drop_duplicates()     
valid = valid.drop_duplicates()
test = test.drop_duplicates()

train['context'] = train['context'].apply(preprocess)
train['question'] = train['question'].apply(preprocess)

valid['context'] = valid['context'].apply(preprocess)
valid['question'] = valid['question'].apply(preprocess)

test['context'] = test['context'].apply(preprocess)
test['question'] = test['question'].apply(preprocess)

#export preprocessed data
train.to_csv('PreprocessedData/train.csv')
valid.to_csv('PreprocessedData/valid.csv')
test.to_csv('PreprocessedData/test.csv')
