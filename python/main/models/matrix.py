from typing import List


class Matrix:

    def __init__(self, matrix_specification: List[List]):
        """
        Initializes a Matrix object given some specification

        :param matrix_specification: list-like of list-likes containing numerics
        """
        self.data = matrix_specification
        assert len(set([len(row) for row in matrix_specification])) == 1

        self.size = (len(self.data), len(self.data[0]))
        self.T = [[self.data[row][col] for row in range(self.size[0])] for col in range(self.size[1])]

    def __eq__(self, other):
        return self.data == other.data

    def __repr__(self):
        out = ["  ".join([str(item) for item in row]) for row in self.data]
        return "\n".join(out)

    def __getitem__(self, index):
        return self.data[index]


class MatrixOperations:

    @staticmethod
    def transpose_matrix(matrix: Matrix):
        """
        Transposes the input matrix by interchanging the rows and columns. Returns the
        transposed matrix.
        """
        m, n = matrix.size

        transposed_matrix = [[matrix.data[row][col] for row in range(m)] for col in range(n)]
        return Matrix(transposed_matrix)
    