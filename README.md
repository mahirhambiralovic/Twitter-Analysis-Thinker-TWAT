# projinda_19

Project by Erik Vanhainen and Mahir Hambiralovic.

# Twitter Analysis Thinker - TWAT
***"Don't worry, be happy"*** - Bob Marley    

In a world of fake news, echo chambers and too much yelling at each other, we've to break through the noise and find positivity. With TWAT you can do just that. TWAT uses the latest, cutting edge, state of the art, ground up, end-to-end, machine learning, AI, machine intelligence statistical models to rate the positivity of a persons messages. Through the website, you will be able to type in a tweet and have the computer tell you if it's positive, neutral or negative.

Inspiration for this project was taken from the following source:
* https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html#sphx-glr-auto-examples-text-plot-document-classification-20newsgroups-py

# How does it work?
## Training algorithm
`scikit-learn` has been the primary library used for the training and prediction of the following algorithm.

Firstly, the algorithm create a bag of words model. The data is first cleaned to make a corpus, where inactive and linking words have been removed as well as smileys and other characters, and the remaining active words have been stemmed (e.g “ran”, “running”, “runs” becomes “run”). The corpus is then transformed into a bag of words, using a ``CountVectorizer``, in which each corpus word is made into a column  in a matrix, and where the individual tweets are the rows. A count of each word in each tweet is then represented in each cell. The column sparcity is set to the 2000 most frequent words.

Finally, TWAT uses a logistic regression model for training and prediction on the bag of words above. Logistic regression is a widely used linear model which uses a logistic function to model a binary dependent variable.

## Dataset
The dataset that we've used is [Sentiment140](http://help.sentiment140.com/for-students/). It contains 1,600,000 classified tweets extracted with the twitter api and contains *target*, *ids*, *date*, *flag*, *user* and *text*. For practicality and due to limited computing power, the training data has been reduced to a random sample of 100,000 tweets.

## Website
Front end website has been made using
* ``flask``
* ``django``

# How to run TWAT
In order to run TWAT the following python libraries are needed:

  * ``pandas``
  * ``numpy``
  * ``nltk``
  * ``sklearn``
  * ``joblib``
  * ``flask``
  * ``giphy``

#### From `src/website/` run the following commands:  
```
export FLASK_APP=application.py
export FLASK_ENV=development
flask run
```

The website will then be accessible from [http://localhost:5000](http://localhost:5000)

![preview](docs/app_preview.gif)
