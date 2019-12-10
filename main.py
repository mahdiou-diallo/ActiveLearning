from sklearn.datasets import fetch_20newsgroups
from active_learning import ActiveLearner, Oracle

# from sklearn.linear_model import LogisticRegression

categories = ['rec.motorcycles', 'rec.sport.baseball',
                'comp.graphics', 'sci.space',
                'talk.politics.mideast']
remove = ("headers", "footers", "quotes")
ng5_train = fetch_20newsgroups(subset='train', categories=categories, remove=remove)
ng5_test = fetch_20newsgroups(subset='test', categories=categories, remove=remove)

# TODO: set up classifier and pass it to the ActiveLearner
# TODO: create Oracle
# ???
# profit