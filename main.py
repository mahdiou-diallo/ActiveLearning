from sklearn.datasets import fetch_20newsgroups
from active_learning import ActiveLearner, Oracle

# from sklearn.linear_model import LogisticRegression

categories = ['rec.motorcycles', 'rec.sport.baseball',
                'comp.graphics', 'sci.space',
                'talk.politics.mideast']
remove = ("headers", "footers", "quotes")
ng5_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
ng5_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

"""
[ (k,ng5_train[k]) for k in ng5_train ]
ng5_train.keys()   ==>  dict_keys(['data', 'filenames', 'target_names', 'target', 'DESCR'])
(ng5_train['data'][1])
"""

## Exploring Data ##
len(ng5_train.data)
len(ng5_test.target)

print("\n".join(ng5_train.data[1].split("\n")))
print(ng5_train.target_names[ng5_train.target[1]])

## Create doc-term Matrix
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(ng5_train.data)
X_train_counts.shape
count_vect.vocabulary_.get(u'algorithm')

## Transform it into frequency matrix aka tf-idf 
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

## Train a Naive Bayes classifier
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, ng5_train.target)

## Create a pipeline to make it simpler
from sklearn.pipeline import Pipeline
text_clf = Pipeline([
     ('vect', CountVectorizer()),
     ('tfidf', TfidfTransformer()),
     ('clf', MultinomialNB()),
])
text_clf.fit(ng5_test.data, ng5_test.target)

## Test on the left documents
import numpy as np
predicted = text_clf.predict(ng5_test.data)
np.mean(predicted == ng5_test.target)

# DONE: set up classifier and pass it to the ActiveLearner
# TODO: create Oracle
# ???
# profit