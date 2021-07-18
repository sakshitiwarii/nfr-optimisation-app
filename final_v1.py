import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from flask import Flask, render_template, url_for, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
import joblib
import numpy as np
import nltk
import os
import nltkmodules
from sklearn.feature_extraction import text
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.datasets import load_diabetes
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# ---------------------------------#
# Page layout
## Page expands to full width
st.set_page_config(page_title='The Machine Learning Hyperparameter Optimization App',
                   layout='wide')

# ---------------------------------#
st.write("""
# The Machine Learning Hyperparameter Optimization App for NFRs
**Please upload your dataset and run the model to predict if your requirement is FR/NFR**

In this implementation, build a model using the **Random Forest** algorithm and search whether your requirement is **Functional** or **Non-Functional**
""")

# ---------------------------------#
# Sidebar - Collects user input features into dataframe
st.sidebar.header('Upload your CSV data')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
st.sidebar.markdown("""
[Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
""")

# Sidebar - Specify parameter settings
st.sidebar.header('Set Parameters')
split_size = st.sidebar.slider('Data split ratio (% for Training Set)', 10, 90, 80, 5)

st.sidebar.subheader('Learning Parameters')
parameter_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 100, 700, (100, 150), 50)
parameter_n_estimators_step = st.sidebar.number_input('Step size for n_estimators', 10)
st.sidebar.write('---')
parameter_max_features = st.sidebar.slider('Max features (max_features)', 1, 50, (1, 3), 1)
st.sidebar.number_input('Step size for max_features', 1)
st.sidebar.write('---')
parameter_min_samples_split = st.sidebar.slider(
    'Minimum number of samples required to split an internal node (min_samples_split)', 1, 10, 2, 1)
parameter_min_samples_leaf = st.sidebar.slider(
    'Minimum number of samples required to be at a leaf node (min_samples_leaf)', 1, 10, 2, 1)

st.sidebar.subheader('General Parameters')
parameter_random_state = st.sidebar.slider('Seed number (random_state)', 1, 1000, 42, 1)
parameter_criterion = st.sidebar.select_slider('Performance measure (criterion)', options=['classification report'])
parameter_bootstrap = st.sidebar.select_slider('Bootstrap samples when building trees (bootstrap)',
                                               options=[True, False])
# parameter_oob_score = st.sidebar.select_slider('Whether to use out-of-bag samples to estimate the R^2 on unseen data (oob_score)', options=[False, True])
parameter_n_jobs = st.sidebar.select_slider('Number of jobs to run in parallel (n_jobs)', options=[1, -1])

n_estimators_range = np.arange(parameter_n_estimators[0], parameter_n_estimators[1] + parameter_n_estimators_step,
                               parameter_n_estimators_step)
max_features_range = np.arange(parameter_max_features[0], parameter_max_features[1] + 1, 1)
param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

# ---------------------------------#
# Main panel

# Displays the dataset
st.subheader('Dataset')


# ---------------------------------#
# Model building

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="model_performance.csv">Download CSV File</a>'
    return href


def build_model(df):
    # X = df['RequirementText'].values.reshape(-1,1)
    df['class'].value_counts()
    df['class'] = np.where(df['class'] == 'F', 1, 0)
    print(df['class'].value_counts())
    X = df.iloc[:, :-1]  # Using all column except for the last column as X
    y = df.iloc[:, -1]  # Selecting the last column as Y
    tweet = df['RequirementText']

    stopwords = stopwords = nltk.corpus.stopwords.words("english")

    other_exclusions = ["#ff", "ff", "rt"]
    stopwords.extend(other_exclusions)

    stemmer = PorterStemmer()

    # stemmer = SnowballStemmer("english")

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
        # tweet = re.sub(r"\S{2,}", " ", tweet)
        # return tweet.strip().lower()

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

    # # Construct tfidf matrix and get relevant scores
    # tfidf = vectorizer.fit_transform(tweet).toarray()
    # vocab = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
    # idf_vals = vectorizer.idf_
    # idf_dict = {i: idf_vals[i] for i in vocab.values()}  # keys are indices; values are IDF scores
    #
    # # Get POS tags for text and save as a string
    # tweet_tags = []
    # for t in tweet:
    #     tokens = basic_tokenize(preprocess(t))
    #     tags = nltk.pos_tag(tokens)
    #     tag_list = [x[1] for x in tags]
    #     tag_str = " ".join(tag_list)
    #     tweet_tags.append(tag_str)
    #
    # # We can use the TFIDF vectorizer to get a token matrix for the POS tags
    # pos_vectorizer = TfidfVectorizer(
    #     tokenizer=None,
    #     lowercase=False,
    #     preprocessor=None,
    #     ngram_range=(1, 3),
    #     stop_words=None,
    #     use_idf=False,
    #     smooth_idf=False,
    #     norm=None,
    #     decode_error='replace',
    #     max_features=5000,
    #     min_df=5,
    #     max_df=0.75,
    # )

    cv = CountVectorizer(max_features=2500)
    X = cv.fit_transform(tweet).toarray()

    st.markdown('A model is being built to predict the following **Y** variable:')
    # st.info(Y.name)

    # Data splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_size)
    # X_train.shape, Y_train.shape
    # X_test.shape, Y_test.shape

    #     rf = RandomForestClassifier(n_estimators=parameter_n_estimators,
    #         random_state=parameter_random_state,
    #         max_features=parameter_max_features,
    #         criterion=parameter_criterion,
    #         min_samples_split=parameter_min_samples_split,
    #         min_samples_leaf=parameter_min_samples_leaf,
    #         bootstrap=parameter_bootstrap,
    #         n_jobs=parameter_n_jobs)

    #     grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    #     grid.fit(X_train, y_train)
    rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score=True)

    param_grid = {
        'n_estimators': [200, 700],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    model = grid_search.fit(X_train, y_train)

    st.subheader('Model Performance')

    Y_pred_test = model.predict(X_test)
    st.write('Coefficient of determination:')
    st.info(classification_report(y_test, Y_pred_test))

    #     st.write('Error (MSE or MAE):')
    #     st.info( mean_squared_error(y_test, Y_pred_test) )

    st.write("The best parameters are %s with a score of %0.2f"
             % (model.best_params_, model.best_score_))

    st.subheader('Model Parameters')
    st.write(model.get_params())

    # -----Process grid data-----#


#     grid_results = pd.concat([pd.DataFrame(grid.cv_results_["params"]),pd.DataFrame(grid.cv_results_["mean_test_score"], columns=["R2"])],axis=1)
#     # Segment data into groups based on the 2 hyperparameters
#     grid_contour = grid_results.groupby(['max_features','n_estimators']).mean()
#     # Pivoting the data
#     grid_reset = grid_contour.reset_index()
#     grid_reset.columns = ['max_features', 'n_estimators', 'R2']
#     grid_pivot = grid_reset.pivot('max_features', 'n_estimators')
#     x = grid_pivot.columns.levels[1].values
#     y = grid_pivot.index.values
#     z = grid_pivot.values

# -----Plot-----#
#     layout = go.Layout(
#             xaxis=go.layout.XAxis(
#               title=go.layout.xaxis.Title(
#               text='n_estimators')
#              ),
#              yaxis=go.layout.YAxis(
#               title=go.layout.yaxis.Title(
#               text='max_features')
#             ) )
#     fig = go.Figure(data= [go.Surface(z=z, y=y, x=x)], layout=layout )
#     fig.update_layout(title='Hyperparameter tuning',
#                       scene = dict(
#                         xaxis_title='n_estimators',
#                         yaxis_title='max_features',
#                         zaxis_title='R2'),
#                       autosize=False,
#                       width=800, height=800,
#                       margin=dict(l=65, r=50, b=65, t=90))
#     st.plotly_chart(fig)

#     #-----Save grid data-----#
#     x = pd.DataFrame(x)
#     y = pd.DataFrame(y)
#     z = pd.DataFrame(z)
#     df = pd.concat([x,y,z], axis=1)
#     st.markdown(filedownload(grid_results), unsafe_allow_html=True)
#     return df
# ---------------------------------#
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df)
    build_model(df)
else:
    st.info('Awaiting for CSV file to be uploaded.')
    # if st.button('Press to use Example Dataset'):
    #     #diabetes = load_diabetes()
    #     df = pd.read_csv('/Users/sakshitiwari/Downloads/redataset.csv', encoding='latin-1')
    #     # X = pd.DataFrame(df.data, columns=diabetes.feature_names)
    #     # Y = pd.Series(diabetes.target, name='response')
    #     # df = pd.concat( [X,Y], axis=1 )
    #     #
    #     # st.markdown('The **Diabetes** dataset is used as the example.')
    #     # st.write(df.head(5))
    #
    # #build_model(df)


# text1 = st.text_area('Enter text')
def predict(text):
    # model = pickle.load(open('nlp_model.pkl', 'rb'))
    # vectorizer = pickle.load(open('transform.pkl', 'rb'))
    # now you can save it to a file
    with open('/Users/sakshitiwari/PycharmProjects/pythonProject3/venv/nlp_model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('/Users/sakshitiwari/PycharmProjects/pythonProject3/venv/transform.pkl', 'rb') as fb:
        vectorizer = pickle.load(fb)

    text = [text]
    vec = vectorizer.transform(text)
    prediction = model.predict(vec)
    result = ''
    if prediction == 1:
        result = 'Functional'
    else:
        result = 'Non-Functional'
    return result


def run():
    #st.sidebar.info('You can either enter the text item online in the textbox or upload a txt file')
    #st.set_option('deprecation.showfileUploaderEncoding', False)
    #add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Txt file"))

    #if add_selectbox == "Online":
    text1 = st.text_area('Enter text')
    output = ""
    if st.button("Predict"):
        output = predict(text1)
        # output = str(output[0]) # since its a list, get the 1st item
        st.success(f"The requirement is {output}")
        st.balloons()
    # elif add_selectbox == "Txt file":
    #     output = ""
    #     file_buffer = st.file_uploader("Upload text file for new item", type=["txt"])
    #     print(file_buffer)
    #     if st.button("Predict"):
    #         text_news = file_buffer.read()
    #
    #     # in the latest stream-lit version ie. 68, we need to explicitly convert bytes to text
    #     st_version = st.__version__  # eg 0.67.0
    #     versions = st_version.split('.')
    #     if int(versions[1]) > 67:
    #         text_news = file_buffer.read()
    #         text_news1 = text_news.decode('utf-8')
    #         print(text_news1)
    #     output = predict(text_news1)
    #     # output = str(output[0])
    #     st.success(f"The news item is {output}")
    #     st.balloons()


if __name__ == "__main__":
    run()

