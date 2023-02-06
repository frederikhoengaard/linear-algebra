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

    def __repr__(self):
        out = ["  ".join([str(item) for item in row]) for row in self.data]
        return "\n".join(out)

    def __getitem__(self, index):
        return self.data[index]
