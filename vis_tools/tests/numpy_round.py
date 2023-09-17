import numpy as np


arr = 5*np.random.random(size=(3,2))

arr_2 = [round(x) for x in arr.round(2).tolist()]

print(arr.round(2).tolist())
print(arr_2)