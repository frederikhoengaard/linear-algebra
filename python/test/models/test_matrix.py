import pytest

from models import Matrix
from models import MatrixOperations


@pytest.mark.parametrize("specification, transpose", [
    ([[1, 2], [3, 4]], [[1, 3], [2, 4]]),
    ([[1, 2], [3, 4], [5, 6]], [[1, 3, 5], [2, 4, 6]])
])
def test_transpose(specification, transpose):
    sample = Matrix(specification)
    transpose = Matrix(transpose)

    assert transpose == MatrixOperations.transpose_matrix(sample)
    assert sample.data == transpose.T
