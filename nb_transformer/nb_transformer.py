import numpy as np
from scipy import sparse
from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)
from sklearn.utils import (
    check_array,
    assert_all_finite
)
from sklearn.utils.validation import check_is_fitted


class NaiveBayesTransformer(BaseEstimator,TransformerMixin):
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
    .. [1] Wang, Sida. Baselines and Bigrams: Simple, Good Sentiment and Topic Classification.
            https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf

    """
    def __init__(self, all_labels, label_of_interest, smoothing_factor=1):
        self.labels = all_labels
        self.label_of_interest = label_of_interest
        self.smoothing_factor = smoothing_factor

    def _get_positive_rows(self):
        """
        Finds all rows of data considered to be the `label_of_interest`

        Returns
        -------
        idxs : array-like
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
        feature_matrix: array-like of shape = [n_samples, n_features]
            Feature matrix representing the vectorized input
            Traditionally from `CountVectorizer` with `binary=True`

        Returns
        -------
        p : array-like of shape = [1, n_features]
            The *unnormalized* conditional probabilities of each word in the "positive" class
        not_p : array-like of shape = [1, n_features]
            The *unnormalized* conditional probabilities of each word in the "negative" class
        """
        feature_matrix = check_array(
            feature_matrix.toarray() if isinstance(feature_matrix, sparse.csr.csr_matrix)
            else feature_matrix
        )
        # get type of feature_matrix
        fm_ = None
        fm_type = None
        fm_shape = None
        if isinstance(feature_matrix, np.ndarray):
            fm_ = "dense"
            fm_type = feature_matrix.dtype
            fm_shape = feature_matrix.shape
        # elif isinstance(feature_matrix, sparse.csr.csr_matrix):
        else:
            fm_ = "sparse"
            fm_type = feature_matrix.dtype
            fm_shape = feature_matrix.get_shape()
        # find indices of p and not_p
        positive_rows = self._get_positive_rows()
        # get summed vectors
        positive_summed = np.zeros((1, fm_shape[1]), dtype=fm_type)
        negative_summed = np.zeros((1, fm_shape[1]), dtype=fm_type)
        for i in range(fm_shape[0]):
            if i in positive_rows:
                if fm_ == "sparse":
                    positive_summed += feature_matrix.getrow(i).toarray()
                elif fm_ == "dense":
                    positive_summed += feature_matrix[i]
            else:
                if fm_ == "sparse":
                    negative_summed += feature_matrix.getrow(i).toarray()
                elif fm_ == "dense":
                    negative_summed += feature_matrix[i]
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
        un_normalized_p: array-like of shape = [1, n_features]
            The *unnormalized* conditional probabilities of each word in the "positive" class
            The [0]th output of `._get_p_not_p()`
        un_normalized_not_p: array-like of shape = [1, n_features]
            The *unnormalized* conditional probabilities of each word in the "negative" class
            The [1]st output of `._get_p_not_p()`

        Returns
        -------
        r : array-like of shape = [1, n_features]
            The resulting ratio of conditional probabilities from Naive Bayes transformation
        """
        un_normalized_p = np.array(un_normalized_p, dtype=np.float32)
        un_normalized_not_p = np.array(un_normalized_not_p, dtype=np.float32)
        # normalize
        p = un_normalized_p / np.sum(un_normalized_p)
        not_p = un_normalized_not_p / np.sum(un_normalized_not_p)
        # calculate r
        r = np.log(p/not_p)
        return r

    def fit(self, X, y=None):
        """Does nothing: this transformer is stateless."""
        X = check_array(
            X.toarray() if isinstance(X, sparse.csr.csr_matrix) else X
        )
        if not isinstance(X, np.ndarray) and not isinstance(X, sparse.csr.csr_matrix):
            raise TypeError(
                "data type of X must be dense or sparse array; type = {}".format(type(X))
            )
        assert_all_finite(X)
        # get type of feature_matrix
        fm_type = None
        if isinstance(X, np.ndarray):
            fm_type = "dense"
        elif isinstance(X, sparse.csr.csr_matrix):
            fm_type = "sparse"
        # get p, not_p
        _p, _not_p = self._get_p_not_p(X)
        # get r
        self._r = self._get_r(_p, _not_p)
        self.X_ = X
        self.y_ = y
        return self

    def transform(self, X, y=None):
        """
        Applies the transformation of features

        Parameters
        ----------
        X: array-like of shape = [n_samples, n_features]
            Feature matrix representing the vectorized input

        Returns
        -------
        transformed_features: array-like of shape = [n_samples, n_features]
            The transformed feature space where word counts have been transformed with
            an elementwise multiplication of the ratio of conditional probabilities from Naive Bayes

        """
        X = check_array(
            X.toarray() if isinstance(X, sparse.csr.csr_matrix) else X
        )
        if not isinstance(X, np.ndarray) and not isinstance(X, sparse.csr.csr_matrix):
            raise TypeError(
                "data type of X must be dense or sparse array; type = {}".format(type(X))
            )
        assert_all_finite(X)
        check_is_fitted(self, ['X_', 'y_'])
        # get type of feature_matrix
        fm_type = None
        if isinstance(X, np.ndarray):
            fm_type = "dense"
        elif isinstance(X, sparse.csr.csr_matrix):
            fm_type = "sparse"
        transformed_features = None
        if fm_type == "sparse":
            transformed_features = X.multiply(self._r)
        elif fm_type == "dense":
            transformed_features = X * self._r
        return transformed_features


    # def fit_transform(self, X, y=None):
    #     """Does nothing: this transformer is stateless."""
    #     X, y = check_X_y(X, y)
    #     assert_all_finite(X)
    #     return self
    #
    # def score(self, X, y=None):
    #     """Does nothing: this transformer is stateless."""
    #     X, y = check_X_y(X, y)
    #     assert_all_finite(X)
    #     return self