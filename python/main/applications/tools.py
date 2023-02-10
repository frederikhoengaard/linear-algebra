from models import Matrix, MatrixOperations


def polynomial_curve_fitting(matrix: Matrix) -> list:
    """
    This function takes a list of nested lists as its input parameter
    where each nested list should have two numeric value entries representing
    an x,y coordinate - read_matrix() can be used to this end.

    The function sorts the input coordinates in ascending order, chooses an n-degree polynomial
    where n is equal to one less than the number of coordinates and creates as system of linear
    equations on the basis of the coordinates and the polynomial as an augmented matrix of the system.

    Solving the equations yields the polynomial function which is returned as a list of coefficients of the form
    p(x) = a_0 + a_1x + a_2x^2 + ... + a_nx^n
    """
    matrix.sort()
    n_coordinates = len(matrix)

    augmented_matrix = []
    for i in range(n_coordinates):
        row = [1]
        x, y = matrix[i]
        for j in range(1, n_coordinates):
            row.append(x**j)
        row.append(y)
        augmented_matrix.append(row)

    augmented_matrix = MatrixOperations.reduced_row_echelon(augmented_matrix)
    return [row[-1] for row in augmented_matrix]


def least_squares_regression(matrix: list) -> list:
    """
    This function takes a list of nested lists as its input parameter
    where each nested list should have two numeric value entries representing
    an x,y coordinate - read_matrix() can be used to this end.

    The function then sorts the coordinates in ascending order and performs the regression,
    returning a list containing the slope a, intercept b and sum of squared errors from the regression line y = b + ax + e
    """
    matrix.sort()

    X = [[1, coordinate[0]] for coordinate in matrix]
    X_transposed = LinAlgToolkit.transpose_matrix(X)

    # Calculate X^T * X
    XTX = LinAlgToolkit.dot_product(X_transposed, X)

    # Calculate X^T * Y
    Y = [[coordinate[1]] for coordinate in matrix]
    XTY = LinAlgToolkit.dot_product(X_transposed, Y)

    # Invert XTX, define slope coefficient a and intercept b
    XTX_inverted = LinAlgToolkit.invert_matrix(XTX)
    results = LinAlgToolkit.tidy_up(LinAlgToolkit.dot_product(XTX_inverted, XTY))
    a = results[1][0]
    b = results[0][0]

    # Calculate sum of squared errors
    error_sum = 0
    for coordinate in matrix:
        error_sum += ((b + coordinate[0] * a) - coordinate[1]) ** 2
    return [b, a, error_sum]


def area_of_triangle(matrix: Matrix) -> float:
    """
    This function takes a 3 x 2 matrix as a list of nested lists representing
    the xy-coordinates of the vertices of a triangle as inpit. It then calculates
    and returns the area of the triangle.
    """
    tmp = []

    for coordinate in matrix:
        tmp.append([coordinate[0], coordinate[1], 1])
    tmp = Matrix(tmp)
    return abs((1 / 2) * MatrixOperations.determinant_rowreduction(tmp))


def volume_of_tetrahedon(matrix: Matrix) -> float:
    """
    This function takes a 4 x 3 matrix as a list of nested lists representing
    the xyz-coordinates of the vertices of a tetrahedon as input. It then
    calculates and returns the volume of the tetrahedon.
    """
    tmp = []

    for coordinate in matrix:
        tmp.append([coordinate[0], coordinate[1], coordinate[2], 1])
    tmp = Matrix(tmp)
    return abs((1 / 6) * MatrixOperations.determinant_rowreduction(tmp))


def test_for_colinearity_xy(matrix: Matrix) -> bool:
    """
    This function takes a 3 x 2 matrix as list of nested lists representing
    three coordinates in the xy-plane as input. It then evalutes and returns
    whether the coordinates are collinear.
    """
    tmp = []

    for coordinate in matrix:
        tmp.append([coordinate[0], coordinate[1], 1])
    tmp = Matrix(tmp)
    return MatrixOperations.determinant_rowreduction(tmp) == 0


def equation_of_line_two_points(matrix: Matrix) -> list:
    """
    This function takes a 2 x 2 matrix as a list of nested lists representing
    two coordinates in the xy-plane as input. It returns the equation for the
    line passing through the two points as a list of the coefficients for
    x,y and b.
    """
    transposed_input = matrix.T
    x = MatrixOperations.determinant_rowreduction(
        [transposed_input[1], [1, 1]]
    )  # TODO: FIX
    y = -1 * MatrixOperations.determinant_rowreduction([transposed_input[0], [1, 1]])
    b = MatrixOperations.determinant_rowreduction(matrix)
    return [x, y, b]
