>>> import numpy as np
>>> x=np.array([1,2,3],[4,5,6])
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
TypeError: Field elements must be 2- or 3-tuples, got '4'
>>> x=np.array([[1,2,3],[4,5,6]])
>>> x.shape
(2, 3)
>>> x=np.array([[11,22,33],[44,55,66]])
>>> x
array([[11, 22, 33],
       [44, 55, 66]])
>>> y=np.array([[1,2,3],[4,5,6]])
>>> np.append(x,y,axis=1)
array([[11, 22, 33,  1,  2,  3],
       [44, 55, 66,  4,  5,  6]])
>>> np.append(x,y,axis=0)
array([[11, 22, 33],
       [44, 55, 66],
       [ 1,  2,  3],
       [ 4,  5,  6]])
>>> np.append(x,y,axis=1).shape
(2, 6)
>>> np.ones((10,1))
array([[1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.],
       [1.]])