import os
import sys
import argparse
import multiprocessing as mp
import time

sys.path.append(os.getcwd() + '/snoop')
sys.path.append(os.getcwd() + '/pyod_detect')
sys.path.append(os.getcwd() + '/lstm_detect')

from top_snoop import TOPSnoop
from detect_onnx import LstmDetect

if __name__=='__main__':
    top_snoop = TOPSnoop()
    lstm_detect = LstmDetect()
    snoop_process = mp.Process(target=top_snoop.run)
    snoop_process.start()
    
