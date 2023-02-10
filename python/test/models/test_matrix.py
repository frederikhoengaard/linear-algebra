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


@pytest.mark.parametrize("matrix_a, matrix_b, expectation", [
    (
            Matrix([[1, 2], [3, 4]]),
            Matrix([[1, 1], [1, 1]]),
            Matrix([[2, 3], [4, 5]])),
    (
            Matrix([[1, 1, 1], [1, 1, 1], [1, 1, 1]]),
            Matrix([[-1, -1, -1], [-1, -1, -1], [-1, -1, -1]]),
            Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]]))
])
def test_add(matrix_a, matrix_b, expectation):
    assert matrix_a + matrix_b == expectation
