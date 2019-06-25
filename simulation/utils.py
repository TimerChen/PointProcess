import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
import os

def dump_all(data, name, dirname = "result"):
    n = len(data)
    for i in range(n):
        json.dump(data[i], open(os.path.join(dirname, name[i]+".json"), "wt"), indent=2)

def load_all(name, dirname="result"):
    n = len(name)
    ret = []
    for i in range(n):
        ret.append(json.load(open(name[i]+".json", "rt")))
    return ret
