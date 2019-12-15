import numpy as np
from sklearn.metrics import get_scorer
from scipy.stats import entropy


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
        """
        super().__init__()
        self.clf = clf

        if strategy == 'least_conf':
            self.uncertainty_scorer = self._confidence_score
        elif strategy == 'margin':
            self.uncertainty_scorer = self._margin_score
        elif strategy == 'entropy':
            self.uncertainty_scorer = self._entropy_score
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
        """Return the entropy of the distribution of class probabilities

        Example
        -------
        >>> probas = np.array([[0., .5, .5],
        ...                    [1., 0., 0.],
        ...                    [.7, .2, .1]])
        >>> np.round(learner._entropy_score(probas), 2)
        array([-0.69, -0.  , -0.8 ])
        """
        return -entropy(probas.T)

    def pick_next_examples(self, X_unlabeled, n):
        """picks the most uncertain examples based on the query strategy
        Parameters:
        ----------
        X_unlabeled : np.ndarray, the unlabeled examples
        n : int, the number of examples to choose

        returns:
        -------
        uncertain_idx: np.ndarray, the indices of the `n` chosen examples
        """
        probas = self.predict_proba(X_unlabeled)
        scores = self.uncertainty_scorer(probas)
        uncertain_idx = np.argsort(-scores)[:n]
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

    def __init__(self, learner: ActiveLearner, metric: str = 'f1_macro'):
        """
        Parameters:
        ----------
        learner : the ActiveLearner to be trained
        metric : the metric that will be tracked,
                can be any valid scikit-learn metric 
        """
        self.learner = learner
        self.scorer = get_scorer(metric)

    def fit(self, X, y, batch_size=None,
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
        if init_labels_idx == 'random':
            init_size = 5 if init_size == None else init_size
            init_labels_idx = np.random.choice(
                y.shape, size=init_size, replace=False)
        bootstrap_idx_[init_labels_idx] = True
        self.learning_examples_ = bootstrap_idx_

        expls = self.learning_examples_
        it = 0
        self.time_chosen_ = np.ones(y.shape, dtype=int) * -1
        self.performance_score_ = []
        while(expls.sum() < expls.shape):
            # training
            L = X[expls, :]
            labels = y[expls]
            self.learner.fit(X=L, y=labels)

            # performance measure
            predictions = self.learner.predict(X)
            self.performance_score_.append(self.scorer(y, predictions))

            # new examples selection
            U = X[~expls, :]
            new_expls = learner.pick_next_examples(X=U, n=self.batch_size_)
            self.time_chosen_[new_expls] = it
            expls[new_expls] = True

            it += 1

    def predict(self, X):
        return self.learner.predict(X)

    def get_choice_distribution(self):
        """Returns at which iteration each example was chosen"""
        return self.time_chosen_

    def get_error_history(self):
        """Returns the performance score of the learner at every iteration"""
        return np.array(self.performance_score_)


if __name__ == "__main__":
    import doctest
    learner = ActiveLearner(None)
    doctest.testmod(extraglobs={'learner': learner})
