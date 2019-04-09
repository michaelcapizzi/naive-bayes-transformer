import numpy as np
from scipy import sparse
from collections import OrderedDict
import warnings
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin
)
from sklearn.utils import (
    check_X_y,
    check_array
)
from copy import deepcopy
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from nb_transformer import NaiveBayesTransformer


# TODO look at http://scikit-learn.org/stable/developers/contributing.html#rolling-your-own-estimator


class NaiveBayesEnhancedClassifier(BaseEstimator, ClassifierMixin):
    """
    Classifier that can utilize the one-v-all required transformations from NaiveBayesTransformer
    Since the NaiveBayesTransformer creates a *different* transformation for each
    label binarization, this class handles the one-v-all approach required to handle multiclass datasets.
    In the case of the a binary dataset, this class simply wraps existing `sklearn` methods
    Default settings apply the best-performing classifier from [1]

    Parameters
    ----------
    base_clf_:
        Any instantiated `sklearn` classifier that
        has either the attribute `.decision_function()` or `.predict_proba()`
    interpolation_factor: float
        The interpolation parameter to mix
        the Multinomial Naive Bayes weights with the Classifier's weights

    Attributes
    ----------

    References
    ----------
    .. [1] Wang, Sida. Baselines and Bigrams: Simple, Good Sentiment and Topic Classification.
            https://nlp.stanford.edu/pubs/sidaw12_simple_sentiment.pdf

    """
    def __init__(self,
                 base_clf=LinearSVC(
                     loss='squared_hinge',
                     penalty='l2',
                     class_weight='balanced',
                     dual=False
                 ),
                 interpolation_factor=0.25
                 ):
        self.base_clf = base_clf
        self.interpolation_factor = interpolation_factor

    # def get_params(self, deep=True):
    #     return self.base_clf.get_params()

    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         self.setattr(parameter, value)
    #     return self

    @staticmethod
    def _build_ovr_classifiers(list_of_all_possible_classes, classifier_instance):
        """
        Builds a binary classifier for each class in a 'one-v-all' approach

        Parameters
        ----------

        list_of_all_possible_classes: list
            All possible classes that appear in the training set
        classifier_instance:
            Any instantiated `sklearn` Classifier

        Returns
        -------
        is_multiclass: bool
            Whether or not the training set is multiclass
        all_clfs: dict
            A <dict> of Classifiers for each label or a single Classifier if not multiclass
        """
        all_clfs = OrderedDict()
        for i in range(len(list_of_all_possible_classes)):
            all_clfs[i] = classifier_instance
        if len(all_clfs) == 2:
            # is just a binary problem!
            del all_clfs[1]
            return False, all_clfs
        elif len(all_clfs) > 2:
            return True, all_clfs
        else:
            warnings.warn("only one label was provided!")
            return False, all_clfs

    @staticmethod
    def _binarize_labels(y):
        """
        Binarizes a set of labels

        Parameters
        ----------
        y: list
            List of all labels for a dataset

        Returns
        -------
        labels_dict: dict
            <dict> of binarized labels for each label in dataset
        """
        distinct_labels = list(set(y))
        all_labels = label_binarize(y, distinct_labels).transpose()
        labels_dict = {}
        for i in range(len(distinct_labels)):
            labels_dict[distinct_labels[i]] = all_labels[i]
        return labels_dict

    def fit(self, X, y):
        """

        Parameters
        ----------
        X: scipy.csr_matrix
            A *sparse* representation of features

        y: array-like of shape = [1, n_samples]
            The labels of the dataset

        Returns
        -------
        self: object
            Returns self

        """
        X, y = check_X_y(
            X.toarray() if isinstance(X, sparse.csr.csr_matrix) else X,
            y
        )
        self.classes_, y = np.unique(y, return_inverse=True)
        clf_copy = deepcopy(self.base_clf)  # needed to pass an estimators
        self.multiclass_, self.ovr_classifiers_ = self._build_ovr_classifiers(
            self.classes_, clf_copy
        )
        self.X_ = X
        self.y_ = y
        self.nb_transformers = {}
        if not self.multiclass_:
            # transform X
            # considering last label in self.classes_ as the `positive`
            self.nb_transformers[0] = NaiveBayesTransformer(y, self.classes_[-1])
            X_transformed = self.nb_transformers[0].fit_transform(X)
            # just handle like a "vanilla" sklearn classifier
            self.ovr_classifiers_[0].fit(X_transformed, y)
            # update weights with interpolation
            try:
                self.ovr_classifiers_[0].coef_ = self._interpolate(self.ovr_classifiers_[0].coef_)
            except:
                warnings.warn(
                    "the classifier you instantiated does not have an attribute `.coef_`; "
                    "interpolation will not occur"
                )
        else:
            # handle with 'one-v-rest' approach
            # binarize labels
            labels_dict = self._binarize_labels(y)
            if labels_dict.keys() != self.ovr_classifiers_.keys():
                raise Exception(
                    "mismatch in labels during fit() and class instantiation; {} != {}".format(
                        labels_dict.keys(), self.ovr_classifiers_.keys()
                    )
                )
            for l, clf in self.ovr_classifiers_.items():
                # transform X for this particular label
                self.nb_transformers[l] = NaiveBayesTransformer(y, l)
                X_transformed = self.nb_transformers[l].fit_transform(X)
                # fit individual classifier with transformed data
                clf.fit(X_transformed, labels_dict[l])
                # update weights with interpolation
                try:
                    clf.coef_ = self._interpolate(clf.coef_)
                except:
                    if l == list(self.ovr_classifiers_.keys())[0]:
                        warnings.warn(
                            "the classifier you instantiated does not have an attribute `.coef_`; "
                            "interpolation will not occur"
                        )
        return self

    def _interpolate(self, coeffs):
        """
        Applies an interpolation  to the learned coefficients of the Classifier

        Parameters
        ----------
        coeffs: array-like of shape = [1, n_features]
            coefficients/weights learned by the Classifier

        Returns
        -------
        interpolated_coeffs: array-like with size of shape = [1, n_features]
            updated coefficients/weights for the Classifier to use in predictions
        """
        if self.interpolation_factor is not None:
            self.interpolation_factor = self.interpolation_factor
        else:
            self.interpolation_factor = 1.0
        # calculate mean magnitude
        mean_magnitude = np.sum(coeffs)/coeffs.shape[1]
        mean_magnitude_vector = np.full(coeffs.shape, mean_magnitude)
        # build interpolated coeffs
        interpolated_coeffs = (1 - self.interpolation_factor) * mean_magnitude_vector + \
                              self.interpolation_factor * coeffs
        return interpolated_coeffs

    def decision_function_predict_proba(self, X):
        """
        Calls either `.decision_function()` or `.predict_proba()` on all Classifiers

        Parameters
        ----------
        X: scipy.csr_matrix
            A *sparse* representation of features

        Returns
        -------
        all_distances: array-like with shape = [n_labels, n_samples]

        """
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(
            X.toarray() if isinstance(X, sparse.csr.csr_matrix) else X
        )
        if not self.multiclass_:
            # transform X
            X_transformed = self.nb_transformers[0].transform(X)
            if hasattr(self.ovr_classifiers_[0], "decision_function"):
                final_distance = self.ovr_classifiers_[0].decision_function(X_transformed)
            else:
                warnings.warn(
                    "classifier you instantiated does not have a method, `decision_function()`; "
                    "using `predict_proba()` instead"
                )
                final_distance = self.ovr_classifiers_[0].predict_proba(X_transformed)
            return final_distance
        else:
            # handle with 'one-v-rest' approach
            all_distances = None
            for l, clf in self.ovr_classifiers_.items():
                # transform X
                X_transformed = self.nb_transformers[l].transform(X)
                if hasattr(clf, "decision_function"):
                    distance = clf.decision_function(X_transformed)
                else:
                    warnings.warn(
                        "classifier you instantiated does not have a method, `decision_function()`;"
                        "using `predict_proba()` instead"
                    )
                    # keep only the probability of the positive class for each datapoint
                    distance = clf.predict_proba(X_transformed)[:, 1].transpose()
                if np.any(all_distances):
                    all_distances = np.vstack((all_distances, distance))
                else:
                    all_distances = distance
            return np.array(all_distances)

    def predict(self, X):
        """
        Calls `.predict()` directly (in the binary-class case) or
        calls `.decision_function()` for each Classifier and then returns label of most confidence

        Parameters
        ----------
        X: scipy.csr_matrix
            A *sparse* representation of features

        Returns
        -------
        labels: array-like with shape = [1, n_samples]
            The predicted label(s)

        """
        if not self.multiclass_:
            # transform X
            X_transformed = self.nb_transformers[0].transform(X)
            return self.ovr_classifiers_[0].predict(X_transformed)
        else:
            # get all boundary distances or probabilities
            all_distances_probs = self.decision_function_predict_proba(X)
            # return most positive (or least negative) margin/probability
            # return [self._index_to_label(idx) for idx in np.argmax(all_distances_probs, axis=0)]
            return self.classes_[np.argmax(all_distances_probs, axis=0)]