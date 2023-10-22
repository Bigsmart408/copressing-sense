import os
import sys
import time
import argparse
import multiprocessing as mp
import subprocess

sys.path.append(os.getcwd() + '/snoop')
sys.path.append(os.getcwd() + '/compressed-sensing')

import runsnoop 
import detect
from recorder import record

if __name__ == '__main__':
    top = runsnoop.TOPSnoop()
    top_process = mp.Process(target=top.run, args=()) 
    top_process.start()
    last = [0, 0, 0]
    time.sleep(500)
    while True:
        a = len(open("csv/all.csv", 'r').readlines())
        detect.detect("csv/vall.csv", "outcome/csv/vall.csv", last[0])
        record("outcome/log/vmem.log", "outcome/csv/vmem.csv", "csv/vmem.csv", last[0])
        last[0] = a
        time.sleep(500)
