

def setup():
    print 1
    
import numpy as np

temp =np.array([[1,2,3,4],[16,18,20,22],[9,8,7,6]])

perm = [10,34,22,8]
i =np.argsort(perm)
i
## [3, 0, 2, 1])
#argsort provides the indexing needed to order from small to laarge
#to reorder the first part of numpy array use below (takes first array then items from each index as listed in i
temp[0] = temp[0,i]

>>> perm = [10,34,22,8]
>>> i =np.argsort(perm)
>>> i
array([3, 0, 2, 1])
