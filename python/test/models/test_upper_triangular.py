import pytest

from models import Matrix
from models import MatrixOperations


@pytest.mark.parametrize("matrix, expected_matrix", [
    (Matrix([[1,2,3],[0,1,-1],[2,2,2]]), Matrix([[1,2,3],[0,1,-1],[0,0,-6]]))
])
def test_make_upper_triangular(matrix, expected_matrix):
    transformed_matrix = MatrixOperations.make_upper_triangular(matrix)
    assert MatrixOperations.is_upper_triangular(transformed_matrix)
    assert transformed_matrix == expected_matrix