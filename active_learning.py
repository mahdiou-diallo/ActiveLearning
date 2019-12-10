import numpy as np
from sklearn.metrics import get_scorer

class ActiveLearner(): # could inherit from some scikit-learn class

    def __init__(self, clf,
                strategy='least_conf'):
        """
        Parameters:
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
        self.learner = learner

        if strategy == 'least_conf':
            self.uncertainty_score = self._confidence_score
        elif strategy == 'margin':
            self.uncertainty_score = self._margin_score
        elif strategy == 'entropy':
            self.uncertainty_score = self._entropy_score
        else:
            raise ValueError(f"Unsupported querying strategy {strategy!r}")

    """Querying strategies"""
    def _confidence_score(self, probas):
        pass

    def _margin_score(self, probas):
        pass

    def _entropy_score(self, probas):
        pass


    def pick_next_examples(self, X_unlabeled, n):
        """picks the examples it wants to learn from based on the query strategy
        Parameters:
        ----------
            X_unlabeled : np.ndarray, the unlabeled examples
            n : int, the number of examples to choose
        
        returns:
        -------
            the indices of the chosen examples
        """
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
    
    def predict_proba(self, X):
        pass


class Oracle():
    """class that knows the labels and can tell them to the `ActiveLearner` when requested
    """
    def __init__(self, learner, metric='f1_macro'):
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

        self.batch_size_ = min(y.shape * .01, 20) if batch_size == None else batch_size

        self.bootstrap_idx_ = np.zeros_like(y, dtype=bool)
        if init_labels_idx == 'random':
            init_size = min(y.shape * .05, 30) if init_size == None else init_size
            init_labels_idx = np.random.choice(y.shape, size=init_size, replace=False)
        self.bootstrap_idx_[init_labels_idx] = True
        self.learning_examples_ = self.bootstrap_idx_

        # TODO: learning loop
        # TODO: create variables and parameters for tracking the classes picked by the learner
        # TODO: define a stopping criterion. it could be user supplied
    
    def predict(self, X):
        pass

    # TODO: add methods for returning the tracked values