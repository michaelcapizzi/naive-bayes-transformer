from sklearn.utils.estimator_checks import check_estimator
from nb_transformer import NaiveBayesTransformer, NaiveBayesEnhancedClassifier


def test_estimator_transformer():
    return check_estimator(NaiveBayesTransformer([0, 1], 1))


def test_estimator_classifier():
    return check_estimator(NaiveBayesEnhancedClassifier())
