from models import Matrix


class LinAlgToolkit:

    @staticmethod   # TODO: move to matrix as dunder
    def matrix_addition(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
        """
        Returns the sum of two equal-sized matrices. If the input parameter matrices are not of equal
        size nothing is returned.
        """
        m_A, n_A = matrix_a.size
        m_B, n_B = matrix_b.size

        if m_A != m_B or n_A != n_B:
            raise MatrixDimensionIncompatibilityError()

        matrix_sum = []
        for i in range(m_A):
            row = []
            for j in range(n_A):
                row.append(matrix_a[i][j] + matrix_b[i][j])
            matrix_sum.append(row)
        return matrix_sum

    @staticmethod  # TODO: move to matrix as dunder
    def matrix_subtraction(matrix_A: Matrix, matrix_B: Matrix) -> Matrix:
        """
        Returns the difference between two equal-sized matrices. If the input parameter matrices are not of equal
        size nothing is returned.
        """
        m_1, n_1 = LinAlgToolkit.get_size(matrix_A)
        m_2, n_2 = LinAlgToolkit.get_size(matrix_B)

        if m_1 != m_2 or n_1 != n_2:
            return

        matrix_difference = []
        for i in range(m_1):
            row = []
            for j in range(n_1):
                row.append(matrix_A[i][j] - matrix_B[i][j])
            matrix_difference.append(row)
        return matrix_difference

    @staticmethod   # TODO: move to matrix as dunder
    def scalar_multiply(matrix: Matrix, scalar: float) -> Matrix:
        """
        Returns input parameter matrix  multiplied by a scalar
        """
        scalar_multiplied_matrix = [row for row in matrix]
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                scalar_multiplied_matrix[i][j] *= scalar
        return scalar_multiplied_matrix


class MatrixDimensionIncompatibilityError(Exception):
    """Raised if operation on matrices is not possible due to dimensions of concerned matrices"""


if __name__ == '__main__':
    pass