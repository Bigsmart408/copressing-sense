from __future__ import print_function
import psutil
from ctypes import c_uint
import os
from os import times
from time import sleep, strftime, time
import argparse
from collections import namedtuple, defaultdict
import psutil
import datetime
import time
import sys
import time


class MemSnoop:

    def main_loop(self, output_filename, interval):
        self.output_file = open(output_filename, 'w')
        self.output_file.write("%s,%s\n"
                               % ("TICKS", "Memoryper%"))
        while True:
            try:
                sleep(interval)
            except KeyboardInterrupt:
                print("receive KeyBoardInterrupt")
                if not self.output_file.closed:
                    self.output_file.flush()
                    self.output_file.close()
                exit()
            cur_time = time.time()
            self.memrecord(cur_time)

    def memrecord(self, time_stamp):
        # cpu逻辑核数
        memory_info = psutil.virtual_memory()

    #总内存
        memory_total = memory_info.total / 1024 / 1024 

    #内存使用率
    #memory_per = (memory_total - memory_info.available) / memory_total * 100
        memory_per = memory_info.percent
        self.output_file.write("%.2f,%.2f\n" % (
            time_stamp, memory_per
        )
                               )
        self.output_file.flush()


if __name__ == "__main__":
    snoop = MemSnoop()
    snoop.main_loop("cpu.csv", 1)
