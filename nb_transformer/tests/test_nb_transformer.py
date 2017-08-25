import numpy as np
from numpy.testing import assert_almost_equal
from scipy import sparse
from nb_transformer import NaiveBayesTransformer

sample_feature_matrix = sparse.csr_matrix(
                            np.array(
                                [[1, 1, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 1, 0],
                                 [0, 1, 0, 1, 0, 0],
                                 [0, 1, 1, 0, 0, 1]]
                            )
                        )

sample_labels = [1, 1, 1, 0]

sample_nb_transform = NaiveBayesTransformer(
    all_labels=[1, 1, 1, 0],
    label_of_interest=1
)


def test_get_positive_rows():
    assert_almost_equal(
        sample_nb_transform._get_positive_rows(),
        np.array([0, 1, 2])
    )


def test_get_p_not_p():
    assert_almost_equal(
        sample_nb_transform._get_p_not_p(sample_feature_matrix),
        (
            np.array([[2.0, 4.0, 1.0, 2.0, 2.0, 1.0]]),
            np.array([[1.0, 2.0, 2.0, 1.0, 1.0, 2.0]])
        )
    )


def test_get_r():
    p, not_p = sample_nb_transform._get_p_not_p(sample_feature_matrix)
    assert_almost_equal(
        sample_nb_transform._get_r(p, not_p),
        np.log(np.array([[1.5, 1.5, 0.375, 1.5, 1.5, 0.375]]))
    )


def test_transform():
    transformed_features = sample_nb_transform.transform(sample_feature_matrix)
    assert_almost_equal(
        transformed_features.toarray(),
        np.array(
            [[0.405, 0.405, 0.0,  0.0,  0.0, 0.0],
             [0.0, 0.405, 0.0, 0.0, 0.405, 0.0],
             [0.0, 0.405, 0.0, 0.405, 0.0, 0.0],
             [0.0, 0.405, -0.98, 0.0, 0.0, -0.98]]
        ),
        decimal=2
    )
