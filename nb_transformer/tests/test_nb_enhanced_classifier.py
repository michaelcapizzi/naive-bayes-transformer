import numpy as np
from numpy.testing import assert_almost_equal
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from nb_transformer import NaiveBayesEnhancedClassifier

sample_feature_matrix = sparse.csr_matrix(
    np.array(
        [[1, 1, 0, 0, 0, 0],
         [0, 1, 0, 0, 1, 0],
         [0, 1, 0, 1, 0, 0],
         [0, 1, 1, 0, 0, 1]]
    )
)

binary_sample_labels = [1, 1, 1, 0]

multiclass_sample_labels = [1, 1, 2, 0]

sample_test_features = sparse.csr_matrix(
    np.array([
        [1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 1, 1]
    ])
)

binary_example = NaiveBayesEnhancedClassifier(interpolation_factor=0.5)

multiclass_example = NaiveBayesEnhancedClassifier([0, 1, 2])


# def test_build_ovr_classifiers_binary():
#     assert len(binary_example.ovr_classifiers) == 1
#
#
# def test_build_ovr_classifiers_multi():
#     assert len(multiclass_example.ovr_classifiers) == 3


def test_binarize_labels():
    dict_ = multiclass_example._binarize_labels([0, 1, 2])
    assert len(dict_) == 3 \
        and 2 not in dict_[0] \
        and 2 not in dict_[1]


def test_interpolate():
    coeffs = np.array([[1, 2, 3]])
    interpolated = binary_example._interpolate(coeffs)
    assert_almost_equal(
        interpolated,
        np.array([[1.5, 2, 2.5]])
    )


def test_decision_function_predict_proba_binary():
    binary_example.fit(sample_feature_matrix, binary_sample_labels)
    decision = binary_example.decision_function_predict_proba(sample_test_features)
    pred = binary_example.predict(sample_test_features)
    for col in range(decision.shape[0]):
        if decision[col] > 0:
            assert pred[col] == 1
        else:
            assert pred[col] == 0


def test_decision_function_predict_proba_multi():
    multiclass_example.fit(sample_feature_matrix, multiclass_sample_labels)
    decisions = multiclass_example.decision_function_predict_proba(sample_test_features)
    preds = multiclass_example.predict(sample_test_features)
    for col in range(decisions.shape[1]):
        assert np.argmax(decisions.transpose()[col]) == preds[col]


def test_predict_binary():
    binary_example.fit(sample_feature_matrix, binary_sample_labels)
    pred = binary_example.predict(sample_test_features)
    assert pred.shape == (2,)


def test_predict_multi():
    multiclass_example.fit(sample_feature_matrix, multiclass_sample_labels)
    pred = multiclass_example.predict(sample_test_features)
    assert pred.shape == (2,)


def test_clf_without_decision_function():
    non_decision_cls = NaiveBayesEnhancedClassifier(
        clf=RandomForestClassifier()
    )
    non_decision_cls.fit(sample_feature_matrix, multiclass_sample_labels)
    decisions = non_decision_cls.decision_function_predict_proba(sample_test_features)
    assert decisions.shape == (2, 3)
