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
    last = [0, 0, 0]
    record("outcome/log/all.log", "outcome/csv/all.csv", "csv/all.csv", last[0])