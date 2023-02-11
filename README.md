# Linear algebra with Python
 
## Description

This repository revolves around linear algebra - matrices in particular. It implements a `Matrix` class on which various operations can be performed e.g. row-echelon reduction, determinant calculation and inversion. 

In practice there is no benefit to using this over say, numpy. It has purely been developed as a tool to verify answers following a typical (under)graduate linear algebra course. The reference literature for this is [Elementary Linear Algebra 8e by Ron Larson](https://www.larsontexts.com/products/title/1).

## Usage

Anything in this library revolves around the [`Matrix`](https://github.com/frederikhoengaard/linear-algebra/blob/main/python/main/models/matrix.py) object. Consider e.g. 

```
>>> list_of_lists = [[1,2,3], [4,5,6], [7,8,9]]

>>> matrix = Matrix(list_of_lists)
```