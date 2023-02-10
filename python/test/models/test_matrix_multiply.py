import pytest

from models import Matrix
from models import MatrixOperations


@pytest.mark.parametrize(
    "matrix_A, matrix_B, expected_result",
    [
        (
            [[2, -1, 3], [5, 1, -2], [2, 2, 3]],
            [[0, 1, 2], [-4, 1, 3], [-4, -1, -2]],
            [[-8, -2, -5], [4, 8, 17], [-20, 1, 4]],
        ),
    ],
)
def test_matrix_multiply(matrix_A, matrix_B, expected_result):
    a = Matrix(matrix_A)
    b = Matrix(matrix_B)
    result = Matrix(expected_result)

    product = MatrixOperations.matrix_multiply(a, b)
    assert result == product
