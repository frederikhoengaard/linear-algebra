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

    @staticmethod
    def is_upper_triangular(matrix: Matrix) -> bool:
        """
        Utility function to test whether a given matrix is upper-triangular.
        Returns true if upper-triangular and false otherwise.
        """
        m, n = matrix.size
        if m != n:
            raise ValueError('Non-square matrices cannot be triangular!')

        for i in range(1, n):
            for k in range(0, i):
                if matrix[i][k] != 0:
                    return False
        return True

    @staticmethod
    def get_rank(matrix: Matrix) -> int:  # TODO: move to matrix
        """
        Rewrites a matrix to row-echelon form, counts the number of non-zero rows
        which is returned as the rank of the matrix.
        """
        reduced_row_echelon_matrix = MatrixOperations.reduced_row_echelon(matrix)
        rank = 0
        for row in reduced_row_echelon_matrix.data:
            for entry in row:
                if entry != 0:
                    rank += 1
                    break
        return rank

    @staticmethod
    def reduced_row_echelon(matrix: Matrix) -> Matrix:
        """

        """
        augmented_matrix = Matrix([row for row in matrix.data])
        m, n = augmented_matrix.size
        n_variables = n - 1
        evaluated_rows = []

        for i in range(n_variables):
            maxrow = 0
            maxval = 0

            for j in range(m):
                if (abs(augmented_matrix.data[j][i]) > abs(maxval)) and j not in evaluated_rows:
                    maxrow = j
                    maxval = augmented_matrix[j][i]
            evaluated_rows.append(maxrow)

            if maxval == 0:
                continue

            other_rows = [row for row in range(m) if row != maxrow]
            reciprocal = 1 / augmented_matrix.data[maxrow][i]
            new_row = [coefficient * reciprocal for coefficient in augmented_matrix.data[maxrow]]

            augmented_matrix.data[maxrow] = new_row

            for row_num in other_rows:
                multiplier = augmented_matrix.data[row_num][i]
                new_other_row = [augmented_matrix.data[row_num][k] - (multiplier * new_row[k]) for k in range(n)]
                augmented_matrix.data[row_num] = new_other_row

        augmented_matrix = MatrixOperations.tidy_up(augmented_matrix)

        if n_variables > m:
            n_exchanges = m
        else:
            n_exchanges = n_variables

        for variable in range(n_exchanges):
            for i in range(m):
                if (augmented_matrix.data[i][variable] != 0) and (i != variable):
                    tmp = augmented_matrix.data[variable]
                    augmented_matrix.data[variable] = augmented_matrix.data[i]
                    augmented_matrix.data[i] = tmp
                    break
        return augmented_matrix

    @staticmethod
    def tidy_up(matrix: Matrix) -> Matrix:  # TODO: move to matrix
        """
        Utility-function to tidy up the contents of a matrix by rounding floats to integers
        where possible or to a maximum of three decimal spaces if value is a floating point.
        Returns the "tidy" matrix.
        """
        tidy_matrix = []
        for row in matrix.data:
            tmp = []
            for value in row:
                if round(value) == round(value, 8):
                    tmp.append(round(value))
                else:
                    tmp.append(round(value, 3))
            tidy_matrix.append(tmp)
        return Matrix(tidy_matrix)

    @staticmethod
    def matrix_multiply(matrix_A: Matrix, matrix_B: Matrix) -> Matrix:
        """
        Returns the product of matrix_A multiplied by matrix_B in this order.
        Returns nothing if product is not defined - that is matrix_A not having
        as many columns as matrix_B has rows.
        """
        _, cols_A = matrix_A.size
        rows_B, cols_B = matrix_B.size

        if cols_A != rows_B:
            raise MatrixDimensionIncompatibilityError

        B_transposed = MatrixOperations.transpose_matrix(matrix_B)

        out = []
        for row in matrix_A:
            tmp = []
            for i in range(cols_B):
                val = 0
                col_to_multiply = B_transposed.data[i]
                for j in range(len(col_to_multiply)):
                    val += row[j] * col_to_multiply[j]
                tmp.append(val)
            out.append(tmp)
        return Matrix(out)


class MatrixDimensionIncompatibilityError(Exception):
    """Raised if operation on matrices is not possible due to dimensions of concerned matrices"""