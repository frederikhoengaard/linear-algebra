from typing import List
import random
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
    infile = open(filename, "r")
    return Matrix([list(map(float, line.split(sep))) for line in infile.readlines()])


def make_system(vars: List[float], random_seed=42):
    random.seed(random_seed)
    var_list = []
    records = {}

    for i in range(1, len(vars) + 1):
        records["x_" + str(i)] = vars[i - 1]

    for key in sorted(records.keys()):
        # print(key,records[key])
        var_list.append((key, records[key]))

    n_equations_to_make = len(vars)

    for j in range(3):
        for i in range(n_equations_to_make):
            total = 0
            eq = []
            for var in var_list:
                variable, value = var
                random_coefficient = random.randint(-3, 4)
                if random_coefficient == 0:
                    continue
                else:
                    total += random_coefficient * value
                    eq.append(str(random_coefficient) + variable)
            eq.append(" = " + str(total))
            print(" ".join(eq))
        print()


if __name__ == "__main__":
    make_system(vars=[1, -1, 3])
