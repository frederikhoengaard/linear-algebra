import pytest

from models import Matrix
from models import MatrixOperations


@pytest.mark.parametrize("specification, expected_result", [
    (
          [
              [1, 1, 1, 1, 26],
              [1, 5, 10, 20, 95],
              [1,-2, 0, 0,-1],
              [0, 1,-4, 0, 0]
          ],
         [
             [1, 0, 0, 0, 15],
             [0, 1, 0, 0, 8],
             [0, 0, 1, 0, 2],
             [0, 0, 0, 1, 1]
         ]
     )
])
def test_row_echelon_reducer(specification, expected_result):
    matrix = Matrix(specification)
    reduced = Matrix(expected_result)
    assert reduced == MatrixOperations.reduced_row_echelon(matrix)