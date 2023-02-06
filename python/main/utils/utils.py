from models import Matrix


def read_csv(filename: str, sep=",") -> Matrix:
    """
    Reads a .csv-file containing separated numeric values e.g. of the form:
    3, 2, 1
    4, 5, 6
    ...
    :param filename: name of file from which to read a matrix
    :param sep: separator to use to parse input file
    :returns: a Matrix object according to the specification
    """
    infile = open(filename, 'r')
    return Matrix([list(map(float, line.split(sep))) for line in infile.readlines()])

