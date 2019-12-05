import numpy as np
import pprint as pp

aaa = np.array([1,2,3,4,5])
print(aaa.shape)
print(aaa)
aaa = aaa.reshape(5,1)
print(aaa.shape)
print(aaa)


bbb = np.array([[1,2,3], [4,5,6]])
print(bbb.shape)

bbb = np.array([[[[1,2,3],[4,5,6]]]])
print(bbb.shape)

ccc = np.array([[1,2],[3,4],[5,6]])
print(ccc.shape)

ddd = ccc.reshape(3,2,1,1)
print(ddd.shape)
pp.pprint(ddd)