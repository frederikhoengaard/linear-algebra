import pytest

from models import Matrix
from models import MatrixOperations


@pytest.mark.parametrize(
    "matrix, determinant",
    [(Matrix([[1, 2, 3], [0, 1, -1], [2, 2, 2]]), -6), (Matrix([[2, -3], [1, 4]]), 11)],
)
def test_determinant(matrix, determinant):
    assert MatrixOperations.det(matrix) == determinant
    assert MatrixOperations.determinant(matrix) == determinant
    assert MatrixOperations.determinant_rowreduction(matrix) == determinant
