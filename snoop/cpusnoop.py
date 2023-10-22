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


class CpuSnoop:

    def main_loop(self, output_filename, interval):
        self.output_file = open(output_filename, 'w')
        self.output_file.write("%s,%s\n"
                               % ("TICKS", "CPU%"))
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
            self.cpurecord(cur_time)

    def cpurecord(self, time_stamp):
        # cpu逻辑核数
        cpu_count = psutil.cpu_count()
        cpu_per = psutil.cpu_percent(interval=0.5, percpu=True)  # 0.5的刷新率
        cpu_per_total = sum(cpu_per, 0)
        cpu_per_format = round(cpu_per_total, 2)
        self.output_file.write("%.2f,%.2f\n" % (
            time_stamp, cpu_per_format
        )
                               )
        self.output_file.flush()


if __name__ == "__main__":
    snoop = CpuSnoop()
    snoop.main_loop("cpu.csv", 1)
