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

    def __add__(self, matrix):
        raise NotImplementedError

    def __sub__(self, matrix):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __eq__(self, matrix):
        return self.data == matrix.data

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
    def matrix_multiply(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
        """
        Returns the product of matrix_A multiplied by matrix_B in this order.
        Returns nothing if product is not defined - that is matrix_A not having
        as many columns as matrix_B has rows.
        """
        _, cols_a = matrix_a.size
        rows_b, cols_b = matrix_b.size

        if cols_a != rows_b:
            raise MatrixDimensionIncompatibilityError

        b_transposed = MatrixOperations.transpose_matrix(matrix_b)

        out = []
        for row in matrix_a:
            tmp = []
            for i in range(cols_b):
                val = 0
                col_to_multiply = b_transposed.data[i]
                for j in range(len(col_to_multiply)):
                    val += row[j] * col_to_multiply[j]
                tmp.append(val)
            out.append(tmp)
        return Matrix(out)

    @staticmethod
    def invert_matrix(matrix: Matrix) -> Matrix:
        """
        Takes a matrix as parameter, returns nothing if parameter matrix is nonsquare and thus non-invertible.
        The function then proceeds to check if the equation system has exactly one solution, and if not it will return
        nothing as the matrix is non-invertible.

        Finally it will adjoin the input parameter matrix with its corresponding identity matrix and reduce it with
        Gauss-Jordan elimination in order to return the inverted matrix.
        """
        m, n = matrix.size
        assert m == n
        identity_matrix = MatrixOperations.create_identity_matrix(m)

        # only matrices with exactly one solution are invertible, so lets use our gauss-jordan script
        reduced_matrix = Matrix([row for row in matrix.data])
        reduced_matrix = MatrixOperations.reduced_row_echelon(reduced_matrix)

        for row in reduced_matrix.data:  # check for zero-row
            invertible = False
            for value in row:
                if value != 0:
                    invertible = True
            if not invertible:
                print('not invertible')
                return

        adjoined_matrix = []
        for i in range(m):  # TODO: make this its own method
            tmp = [val for val in matrix.data[i]]
            tmp.extend(identity_matrix.data[i])
            adjoined_matrix.append(tmp)

        adjoined_matrix = Matrix(adjoined_matrix)

        reduced_adjoined_matrix = MatrixOperations.reduced_row_echelon(adjoined_matrix)
        inverted_matrix = []
        for i in range(m):
            inverted_matrix.append([reduced_adjoined_matrix.data[i][j] for j in range(n, (2 * n))])
        return Matrix(inverted_matrix)

    @staticmethod
    def create_identity_matrix(order: int) -> Matrix:
        """
        Returns the identity matrix of a given order as a list of lists.
        """
        out = []
        for i in range(order):
            tmp = []
            for j in range(order):
                if i == j:
                    tmp.append(1)
                else:
                    tmp.append(0)
            out.append(tmp)
        return Matrix(out)

    @staticmethod
    def determinant(matrix: Matrix) -> float:
        """
        This function takes an n x n matrix as a list of nested lists as input. It then
        calculates and returns its determinant by use of cofactor expansion.
        """
        m,n = matrix.size

        if m != n:
            raise ValueError("Non-square matrices do not have determinants!")

        else:
            if m == 1 and n == 1:
                return matrix.data[0][0]

            elif m == 2:
                val = matrix.data[0][0] * matrix.data[1][1] - matrix.data[0][1] * matrix.data[1][0]
                return val

            else:
                det = 0
                new_matrix = Matrix([row for row in matrix[1:]])
                new_matrix = MatrixOperations.transpose_matrix(new_matrix)
                for i in range(n):
                    pivot = matrix[0][i]
                    cols_to_select = [j for j in range(n) if j != i]
                    minor_matrix = MatrixOperations.transpose_matrix(Matrix([new_matrix[col] for col in cols_to_select]))

                    if i % 2 == 0: # cofactor is positive
                        det += pivot * MatrixOperations.determinant(minor_matrix)
                    else: # cofactor is negative
                        det -= pivot * MatrixOperations.determinant(minor_matrix)
                return det

    @staticmethod
    def determinant_rowreduction(matrix: Matrix) -> float:
        """
        This function takes an n x n matrix as a list of nested lists as input.

        It returns the determinant of the triangular matrix calculated as the product
        of the elements of its main diagonal.
        """
        tmp = Matrix([row for row in matrix])
        m, n = matrix.size

        if m != n:
            raise ValueError("Non-square matrices do not have determinants!")

        else:
            for i in range(n - 1):
                var = tmp.data[i][i]
                if var != 0:
                    for j in range(i + 1, n):
                        multiplier = tmp.data[j][i] / var
                        for k in range(n):
                            tmp.data[j][k] -= (multiplier * tmp.data[i][k])
            det = 1
            for i in range(n):
                det *= tmp.data[i][i]
            return det


class MatrixDimensionIncompatibilityError(Exception):
    """Raised if operation on matrices is not possible due to dimensions of concerned matrices"""


if __name__ == '__main__':
    matrix = Matrix([[1,2,3],[0,1,-1],[2,2,2]])
    inverse = MatrixOperations.invert_matrix(matrix)
    print(inverse.__repr__())

    print(MatrixOperations.matrix_multiply(inverse, matrix).__repr__())

