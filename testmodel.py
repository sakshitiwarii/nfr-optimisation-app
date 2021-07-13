from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import numpy as np
import nltk
import os
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('/Users/sakshitiwari/Downloads/redataset.csv', encoding='latin-1')

df['class'].value_counts()

df['class'] = np.where(df['class'] == 'F', 1, 0)

X = df['RequirementText']
y = df['class']

y.head()

df['class'].hist()

tweet = df['RequirementText']

stopwords = stopwords = nltk.corpus.stopwords.words("english")

other_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(other_exclusions)

stemmer = PorterStemmer()

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                       '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    return parsed_text


def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""

    tweet = tweet.strip().strip('"')
    tweet = re.sub(r'[^A-Za-z0-9(),!?\.\'\`]', ' ', tweet)
    tweet = re.sub(r"\'s", " \'s", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet)
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"\.", "  \. ", tweet)
    tweet = re.sub(r"\"", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)

    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""

    tweet = tweet.strip().strip('"')
    tweet = re.sub(r'[^A-Za-z0-9(),!?\.\'\`]', ' ', tweet)
    tweet = re.sub(r"\'s", " \'s", tweet)
    tweet = re.sub(r"\'ve", " \'ve", tweet)
    tweet = re.sub(r"n\'t", " n\'t", tweet)
    tweet = re.sub(r"\'re", " \'re", tweet)
    tweet = re.sub(r"\'d", " \'d", tweet)
    tweet = re.sub(r"\'ll", " \'ll", tweet)
    tweet = re.sub(r",", " , ", tweet)
    tweet = re.sub(r"\.", "  \. ", tweet)
    tweet = re.sub(r"\"", " , ", tweet)
    tweet = re.sub(r"!", " ! ", tweet)
    tweet = re.sub(r"\(", " \( ", tweet)
    tweet = re.sub(r"\)", " \) ", tweet)
    tweet = re.sub(r"\?", " \? ", tweet)
    # tweet = re.sub(r"\S{2,}", " ", tweet)
    # return tweet.strip().lower()

    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()

vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.75
)

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(tweet).toarray()
pickle.dump(cv, open('transform.pkl', 'wb'))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier(n_jobs=-1,max_features= 'sqrt' ,n_estimators=50, oob_score = True)

param_grid = {
    'n_estimators': [200, 700],
    'max_features': ['auto', 'sqrt', 'log2']
}

grid_search = GridSearchCV(estimator=rfc, param_grid = param_grid, cv=5, verbose=2)

model = grid_search.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_preds = model.predict(X_test)

report = classification_report(y_test, y_preds)

print(report)

a = accuracy_score(y_test, y_preds)

print(a)

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_preds)

clf=model.fit(X_train, y_train)
filename = 'nlp_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(clf, file)



