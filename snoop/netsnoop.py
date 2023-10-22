from __future__ import print_function
import threading
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


class NetSnoop:

    def mainloop(self, output_filename, interval):
        self.output_file = open(output_filename, 'w')
        self.output_file.write("%s,%s,%s\n"
                               % ("TICKS", "Net_recv", "Net_sent"))
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
            self.netrecord(cur_time)

    def netrecord(self, time_stamp):
        net = psutil.net_io_counters()
        net_recv = float(net.bytes_recv / 1024 / 1024)
        # 发送数据
        net_sent = float(net.bytes_sent / 1024 / 1024)

        self.output_file.write("%.2f,%.2f,%.2f\n" % (
            time_stamp,
            net_recv,
            net_sent
        )
                               )
        self.output_file.flush()


if __name__ == "__main__":
    snoop = NetSnoop()
    snoop.mainloop("network.csv", 1)
