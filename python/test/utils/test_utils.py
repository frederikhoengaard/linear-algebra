from utils import read_csv


def test_read_csv():
    matrix = read_csv("assets/sample_3x4_matrix.csv")
    assert matrix.size == (3, 4)
