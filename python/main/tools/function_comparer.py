# function_comparer.py

import os
import lin_alg_toolkit
from time import time

current_dir = os.getcwd()
test_dir_path = current_dir + '/test_data/test_determinant/'
test_files = os.listdir(test_dir_path)


def test_function(function_name, test_data):
    print('Testing function:',function_name.__name__, '...')
    start_time = time()
    result = function_name(test_data)
    end_time = time()

    print(function_name.__name__,'returned',result,'in',str(end_time-start_time),'s')
    return result



for filename in test_files:
    path = ''.join([test_dir_path, filename]) 
    matrix = lin_alg_toolkit.read_matrix(path)

    print('Evaluating file',path,'\n')
    print('Matrix is:\n')
    for line in matrix:
        print(line)

    print('\n')
    A = test_function(lin_alg_toolkit.determinant,matrix)

    print('\n')
    B = test_function(lin_alg_toolkit.determinant_rowreduction,matrix)

    if A != B:
        print('ERROR: Functions returning different results!')

    print('\n ==============================================')