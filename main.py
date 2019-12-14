from sklearn.datasets import fetch_20newsgroups
from active_learning import ActiveLearner, Oracle

# from sklearn.linear_model import LogisticRegression

categories = ['rec.motorcycles', 'rec.sport.baseball',
                'comp.graphics', 'sci.space',
                'talk.politics.mideast']
remove = ("headers", "footers", "quotes")
ng5_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
ng5_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

## Import unlabeled data => Classic3
f = open('/Users/amine/Desktop/text-AN/project/classic3_raw.txt', 'r')
classic3 = f.readlines()
f.close()

## Exploring Data ##
len(ng5_train.data)
len(ng5_test.target)

print("\n".join(ng5_train.data[1].split("\n")))
print(ng5_train.target_names[ng5_train.target[1]])
"""
[ (k,ng5_train[k]) for k in ng5_train ]
ng5_train.keys()   ==>  dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
(ng5_train['data'][1])
"""

##  Third step : Select n samples from non labeled data U  ##
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

## We can eather call functions separately or call the following pipeline
""" 
## Create doc-term Matrix
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(ng5_train.data)
X_train_counts.shape
count_vect.vocabulary_.get(u'algorithm')

## Transform it into frequency matrix aka tf-idf 
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

## Train a Naive Bayes classifier
clf = MultinomialNB().fit(X_train_tfidf, ng5_train.target)
"""

## Create a pipeline to make it simpler
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
])
text_clf.fit(ng5_test.data, ng5_test.target)

## Predict U data
import numpy as np
# number of samples
n = 100

# predicted classes and corresponding probabilities
predicted = text_clf.predict(ng5_test.data)
predicted_proba = text_clf.predict_proba(ng5_test.data)
#np.mean(predicted == ng5_test.target)

## Least Confidence (aka. Uncertainty) Strategy
uncertainty = 1 - predicted_proba.max(axis=1)
uncertainty.size

# index of top n uncertainty score
ind = np.argpartition(uncertainty, -n)[-n:]
uncertainty[ind]

## Margin Sampling
part = np.partition(-predicted_proba, 1, axis=1)
margin = - part[:, 0] + part[:, 1]

# index of n min margin score
ind = np.argpartition(margin, n)[:n]
margin[ind]

## Entropy based
from scipy.stats import entropy

entropy = entropy(predicted_proba.T)
# index of top n entropy score
ind = np.argpartition(entropy, -n)[-n:]
entropy[ind]

# IN_PROGRESS : set up classifier and pass it to the ActiveLearner
# TODO: create Oracle
# ???
# profit