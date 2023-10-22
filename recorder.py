import os
import sys
import time
import pandas as pd

def record(save, predict, value, start):
    localtime = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    df = pd.read_csv(value)

    f = open(predict)
    out = open(save, 'a')
    offset = 0
    line = f.readline()
    num =0
    while line:
        
        if float(line) > 0.0 :
            out.write(localtime + ":\n")
            out.write(str(df.loc[offset + start].round(2).apply('{:.2f}'.format)))
            out.write("\n\n")
        offset = offset + 1
        line = f.readline()
