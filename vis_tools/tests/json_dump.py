import json
import numpy as np

lst = [1, 2, 3]

json_string = json.dumps(lst)

with open('test.json', 'w') as outfile:
    json.dump(lst, outfile)

