import numpy as np
from scipy import sparse
from sklearn.base import (BaseEstimator, TransformerMixin)


class NaiveBayesTransformer(BaseEstimator, TransformerMixin):
    """
    Feature transformation that uses the conditional probabilities from
    Naive Bayes as the feature values.

    Parameters
    ----------
    all_labels: list
        A <list> of labels for all data points in the training set
    label_of_interest: int
        The label considered to be "positive"
        Note: This transformation results in *different* values for each label
    smoothing_factor: float
        Smoothing value to be used when estimating conditional probabilities

    Attributes
    ----------
    _r : np.array
        The resulting ratio of conditional probabilities from Naive Bayes transformation
        which is applied in an elementwise multiplication to the existing feature matrix

    References
    ----------
    .. [1] Wang, Sida. Baselines and Bigrams: Simple, Good Sentiment and Topic Classification. https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf

    """
    def __init__(self, all_labels, label_of_interest, smoothing_factor=1):
        self.labels = all_labels
        self.label_of_interest = label_of_interest
        self.smoothing_factor = smoothing_factor
        self._r = None

    def _get_positive_rows(self):
        """
        Finds all rows of data considered to be the `label_of_interest`

        Returns
        -------
        idxs : np.array
            The array of indexes from self.labels that are "positive"
        """
        # ensure labels as np array
        try:
            as_array = np.array(self.labels)
        except:
            as_array = self.labels
        # get indices of label_of_interest
        idxs = np.where(as_array == self.label_of_interest)[0]
        return idxs

    def _get_p_not_p(self, feature_matrix):
        """
        Builds the *unnormalized* `p` and `not_p` vector

        Parameters
        ----------
        feature_matrix: scipy.csr.sparse.matrix
            Feature matrix representing the vectorized input
            Traditionally from `CountVectorizer` with `binary=True`

        Returns
        -------
        p : np.array
            The *unnormalized* conditional probabilities of each word in the "positive" class
        not_p : np.array
            The *unnormalized* conditional probabilities of each word in the "negative" class
        """
        # find indices of p and not_p
        positive_rows = self._get_positive_rows()
        # get summed vectors
        positive_summed = np.zeros((1, feature_matrix.get_shape()[1]))
        negative_summed = np.zeros((1, feature_matrix.get_shape()[1]))
        for i in range(feature_matrix.get_shape()[0]):
            if i in positive_rows:
                positive_summed += feature_matrix.getrow(i).toarray()
            else:
                negative_summed += feature_matrix.getrow(i).toarray()
        # smooth
        p = positive_summed + self.smoothing_factor
        not_p = negative_summed + self.smoothing_factor
        return p, not_p

    @staticmethod
    def _get_r(un_normalized_p, un_normalized_not_p):
        """
        Builds the `r` vector representing the `log` ratio of *normalized* `p` to `not_p`

        Parameters
        ----------
        un_normalized_p: np.array
            The *unnormalized* conditional probabilities of each word in the "positive" class
            The [0]th output of `._get_p_not_p()`
        un_normalized_not_p: np.array
            The *unnormalized* conditional probabilities of each word in the "negative" class
            The [1]st output of `._get_p_not_p()`

        Returns
        -------
        r : np.array
            The resulting ratio of conditional probabilities from Naive Bayes transformation
        """
        # normalize
        p = un_normalized_p / np.sum(un_normalized_p)
        not_p = un_normalized_not_p / np.sum(un_normalized_not_p)
        # calculate r
        r = np.log(p/not_p)
        return r

    def transform(self, X, *_):
        """
        Applies the transformation of features

        Parameters
        ----------
        X: scipy.csr.sparse.matrix
            Feature matrix representing the vectorized input

        Returns
        -------
        transformed_features: np.array
            The transformed feature space where word counts have been transformed with
            an elementwise multiplication of the ratio of conditional probabilities from Naive Bayes

        """
        if not np.any(self._r):
            # this is being called during training, so must build self._r
            # get p, not_p
            _p, _not_p = self._get_p_not_p(X)
            # get r
            self._r = self._get_r(_p, _not_p)
        transformed_features = sparse.csr_matrix(X.multiply(self._r))
        return transformed_features

    def fit(self, *_):
        return self
