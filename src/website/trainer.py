import pandas
import re
from time import time
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import ssl
import random
from joblib import dump, load

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

key_pos = 0
text_pos = 5


# Training method.
def train(training_sample_size, trainfile, testfile):
    # Load train and test data
    with open(trainfile) as f:
        file_size = sum(1 for line in f)
    data_train, data_test = load_data(training_sample_size, file_size, trainfile, testfile)
    print("Extracting features from the training data using a sparse vectorizer")
    t0 = time()
    corpus = corpusGenerator(data_train)

    # Vectorize, reduce sparcity with only 2000 most frequent words
    cv = CountVectorizer(max_features=2000)

    # Matrix of word tokens in alphabetical order as rows and tweets as columns, with count as value.
    x_train = cv.fit_transform(corpus).toarray()
    y_train = data_train.iloc[:,key_pos].values
    duration = time() - t0
    print("done in %fs" % (duration))
    print("n_samples: %d, n_features: %d" % x_train.shape)
    print()
    print("Extracting features from the test data using the same vectorizer")
    t0 = time()
    corpus = corpusGenerator(data_test)

    # Vectorize
    x_test = cv.transform(corpus).toarray()
    y_test = data_test.iloc[:,key_pos].values
    duration = time() - t0
    print("done in %fs" % (duration))
    print("n_samples: %d, n_features: %d" % x_test.shape)
    print()

    # Logistic Regression
    log_model = LogisticRegression()
    log_model = log_model.fit(X=x_train, y=y_train)
    y_prediction = log_model.predict(x_test)

    # Generate Confusion Matrix
    cm = confusion_matrix(y_test,y_prediction)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Print Confusion Matrix
    print('Normalized confusion matrix with Logistic regression')
    print("Act     Pred\n       N       P\nN     %.2f  ,  %.2f\nP     %.2f  ,  %.2f"%(cm_normalized[0,0], cm_normalized[2,0],cm_normalized[0,2], cm_normalized[2,2]))

    return cv, log_model

# Saving models for persistance
cv_filename = '../../data/learning_objects/cv.joblib'
log_model_filename = '../../data/learning_objects/log_model.joblib'

def save_obj_files(cv,log_model):
    dump(cv, cv_filename)
    dump(log_model, log_model_filename)

def load_obj_files():
    cv = load(cv_filename)
    log_model = load(log_model_filename)
    return cv, log_model


# Load data.
# Sentiment labels are:
# Training: (0=negative, 4=positive)
# Test: (0=negative, 2=neutral 4=positive)
def load_data(training_sample_size, file_size, trainfile, testfile):
        skip = sorted(random.sample(range(0,file_size),file_size-training_sample_size))
        data_train = pandas.read_csv(trainfile, header=None, usecols=[key_pos,text_pos], skiprows=skip, encoding='latin-1')
        data_test = pandas.read_csv(testfile, header=None, usecols=[key_pos,text_pos], encoding='latin-1')
        print('data loaded')
        print("data_train   - %d documents" % (len(data_train)))
        print("data_test    - %d documents" % (len(data_test)))
        print()
        return data_train, data_test


# Cleans data by stemming and porterstems. data = string or array of strings
def corpusGenerator(data):
    corpus = []
    if type(data) is pandas.core.frame.DataFrame:
        for j in range(0,len(data)):
            text = re.sub('\W', ' ', data[text_pos][j])
            text = text.lower()
            text = text.split()

            # Takes word and breaks to its core component (e.g. run,ran,running to run)
            ps = PorterStemmer()

            # Remove all stopwords and porterstem them
            text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
            text = ' '.join(text)
            corpus.append(text)

    elif type(data) is str:
        text = re.sub('\W', ' ', data)
        text = text.lower()
        text = text.split()

        # Takes word and breaks to its core component (e.g. run,ran,running to run)
        ps = PorterStemmer()

        # Remove all stopwords and porterstem them
        text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]
        text = ' '.join(text)
        corpus.append(text)
    return corpus

# Prediction
def predict(user_input, cv, log_model):
    corpus = corpusGenerator(user_input)
    x_test_new = cv.transform(corpus).toarray()
    prediction_prob = log_model.predict_proba(x_test_new)[0,1]
    print(prediction_prob)

    # Predictiion probabilities of 0.4<x<0.6 are considered neutral
    if prediction_prob < 0.4:
        sentiment = "Negative dude..."
    elif prediction_prob > 0.6:
        sentiment = "Positive vibezzz"
    else:
        sentiment = "Neutral I guess."
    return sentiment
