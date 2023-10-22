from __future__ import print_function
import psutil
from ctypes import c_uint
import os
from os import times
from bcc import BPF
from time import sleep, strftime, time
import argparse
from collections import namedtuple, defaultdict
import psutil
import datetime
import time
import sys
import time
import libvirt
from xml.etree import ElementTree


class VcpuSnoop:

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
        conn = libvirt.open("qemu:///system")  # 打开本地虚拟机信息
        for id in conn.listDomainsID():
            domain = conn.lookupByID(id)
            t1 = time.time()  # 可以使用time.time_ns()将数据精确到ns级别
            c1 = int(domain.info()[4])
            time.sleep(1)  # 时间间隔尽量小
            t2 = time.time()
            c2 = int(domain.info()[4])
            c_nums = int(domain.info()[3])
            cpu_per = (c2 - c1) * 100 / ((t2 - t1) * c_nums * 1e9)
            self.output_file.write("%.2f,%.2f\n" % (
                time_stamp, cpu_per
            )
                                   )
            self.output_file.flush()


if __name__ == "__main__":
    snoop = VcpuSnoop()
    snoop.main_loop("vcpu.csv", 1)
