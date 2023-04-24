import numpy as np

def diff_arr(x, i, j)
    x1 = x.reshape((i,j))
    print(xl)    
    diff= x1.max(axis=1) - x1.min(axis=1)
    return diff

 #use the function: 
 x = np.arange(12)
 print(x)
 
 diff = lb.diff_arr(x,2,6)
 print("\nDifference between the maximum and the minimum values of the said array:")
 print(diff)
