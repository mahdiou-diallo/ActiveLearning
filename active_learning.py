import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.stats import entropy
from types import FunctionType
from sklearn.base import clone


class ActiveLearner(object):  # could inherit from some scikit-learn class

    def __init__(self, clf, strategy='least_conf'):
        """
        Parameters
        ----------
        clf : classifier, any classifier with a fit, predict, and predict_proba methods
            can be any scikit-learn classifier
        strategy : str, the querying strategy
            `'least_conf'`: least confidence query, the examples with the lowest max proba are chosen
            `'margin'` : margin query, the examples with the lowest difference
                        between most probable and second most probable are chosen
            `'entropy'` : entropy query, the examples with the highest entropy are chosen
            `'random'` : random examples are chosen
        """
        super().__init__()
        self.clf = clf

        if strategy == 'least_conf':
            self.uncertainty_scorer = self._confidence_score
        elif strategy == 'margin':
            self.uncertainty_scorer = self._margin_score
        elif strategy == 'entropy':
            self.uncertainty_scorer = self._entropy_score
        elif strategy == 'random':
            self.uncertainty_scorer = self._random_score
        else:
            raise ValueError(f"Unsupported querying strategy {strategy!r}")

    """Querying strategies"""

    def _confidence_score(self, probas: np.ndarray):
        """Return the probability of the most likely class
        Example
        -------
        >>> probas = np.array([[0., .5, .5],
        ...                    [1., 0., 0.],
        ...                    [.7, .2, .1]])
        >>> learner._confidence_score(probas)
        array([0.5, 1. , 0.7])
        """
        return probas.max(axis=1)

    def _margin_score(self, probas: np.ndarray):
        """Return the margin between the two most likely classes

        Example
        -------
        >>> probas = np.array([[0., .5, .5],
        ...                    [1., 0., 0.],
        ...                    [.7, .2, .1]])
        >>> learner._margin_score(probas)
        array([0. , 1. , 0.5])
        """
        probas_sorted = np.sort(probas, axis=1)
        margin = probas_sorted[:, -1] - probas_sorted[:, -2]
        return margin

    def _entropy_score(self, probas: np.ndarray):
        """Return 1 minus the entropy of the distribution of class probabilities

        Example
        -------
        >>> probas = np.array([[0., .5, .5],
        ...                    [1., 0., 0.],
        ...                    [.7, .2, .1]])
        >>> np.round(learner._entropy_score(probas), 2)
        array([0.31, 1.  , 0.2 ])
        """
        ent = entropy(probas.T)  # calculate entropy
        ent = ent.max() - ent  # make zero the minimum
        return ent / ent.max()  # scale it to be in the [0, 1] range

    def _random_score(self, probas: np.ndarray):
        return 1-np.random.uniform(size=probas.shape[0])

    def pick_next_examples(self, X_unlabeled, n):
        """picks the most uncertain examples based on the query strategy
        Parameters
        ----------
        X_unlabeled : np.ndarray, the unlabeled examples
        n : int, the number of examples to choose

        returns
        -------
        uncertain_idx: np.ndarray, the indices of the `n` chosen examples
        """
        m = X_unlabeled.shape[0]
        if m <= n:
            return np.arange(m)

        probas = self.predict_proba(X_unlabeled)
        scores = self.uncertainty_scorer(probas)
        uncertain_idx = np.argpartition(scores, n)[:n]
        return uncertain_idx

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def predict(self, X):
        return self.clf.predict(X)

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class Oracle(object):
    """class that knows the labels and can provide them to the `ActiveLearner` when requested
    """

    def __init__(self, learner: ActiveLearner, metrics=[f1_score]):
        """
        Parameters:
        ----------
        learner : the ActiveLearner to be trained
        metrics : function or list[function], the metric (or list of metrics) that will be tracked,
                they can be any valid scikit-learn metrics
        """
        self.learner = learner

        if isinstance(metrics, list):
            self.scorers = metrics
        elif isinstance(metrics, FunctionType):
            self.scorers = [metrics]
        else:
            raise ValueError(f"Unsupported metric type type {type(metrics)!r}")

        # for s in self.scorers:
        #     print(type(s))

    def fit(self, X, y, X_test, y_test, batch_size=None,
            init_size=None, init_labels_idx='random'):
        """train the `ActiveLearner` and keep track of the metrics
        Parameters:
        ----------
        X : np.ndarray, the attributes
        y : np.ndarray, the labels
        batch_size : int, the number of new examples at each iteration
        init_size : int, the number of initial examples the learner can receive
        init_labels : np.ndarray[int], the indices of the initial examples
        """

        self.batch_size_ = 1 if batch_size == None else batch_size

        bootstrap_idx_ = np.zeros_like(y, dtype=bool)
        if isinstance(init_labels_idx, str) and init_labels_idx == 'random':
            init_size = 5 if init_size == None else init_size
            init_labels_idx = np.random.choice(
                y.shape[0], size=init_size, replace=False)
        bootstrap_idx_[init_labels_idx] = True
        learning_examples = bootstrap_idx_

        it = 0
        # we put -1 to mark the initial examples
        self.time_chosen_ = np.ones(y.shape, dtype=int) * -1
        self.performance_scores_ = []
        self.models_ = []

        while(learning_examples.sum() < learning_examples.shape[0]):
            # training
            L = X[learning_examples, :]
            labels = y[learning_examples]
            # pas cool mais ca marche
            self.learner.fit(X=L, y=labels)

            # save model
            self.models_.append(clone(self.learner, safe=False))

            # performance measure
            predictions = self.learner.predict(X_test)
            self.performance_scores_.append(
                [scorer(y_test, predictions, average='micro') for scorer in self.scorers])

            # new examples selection
            U = X[~learning_examples, :]
            new_expls = self.learner.pick_next_examples(
                U, n=self.batch_size_)

            u_idx, = np.where(~learning_examples)
            chosen_idx = u_idx[new_expls]
            self.time_chosen_[chosen_idx] = it
            learning_examples[chosen_idx] = True

            it += 1

        columns = [s.__name__ for s in self.scorers]
        self.performance_scores_ = pd.DataFrame(
            self.performance_scores_, columns=columns)

    def predict(self, X):
        return self.learner.predict(X)


def run_doctests():
    import doctest
    learner = ActiveLearner(None)
    doctest.testmod(extraglobs={'learner': learner})


if __name__ == "__main__":
    run_doctests()
