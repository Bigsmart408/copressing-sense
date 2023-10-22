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


class AllSnoop:

    def main_loop(self, output_filename, interval):
        self.output_file = open(output_filename, 'w')
        self.output_file.write("%s,%s,%s,%s,%s\n"
                               % ("TICKS", "CPU%", "Memper%", "Net_recv", "Net_sent"))
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
            self.allrecord(cur_time)

    def allrecord(self, time_stamp):

        # cpu的使用率
        cpu_count = psutil.cpu_count()
        cpu_per = psutil.cpu_percent(interval=0.5, percpu=True)  # 0.5的刷新率
        cpu_per_total = sum(cpu_per, 0)
        cpu_per_format = round(cpu_per_total, 2)

        # 内存信息
        memory_info = psutil.virtual_memory()

        # 总内存
        memory_total = memory_info.total / 1024 / 1024

        # 内存使用率
        # memory_per = (memory_total - memory_info.available) / memory_total * 100
        memory_per = memory_info.percent

        # 硬盘信息
        disk_info = psutil.disk_usage("/")  # 根目录磁盘信息

        # print(disk_ingo)
        # 根目录大小
        disk_total = disk_info.total

        # 根目录使用情况
        disk_per = float(disk_info.used / disk_total * 100)

        # 网络使用情况
        net = psutil.net_io_counters()

        # 收取数据
        net_recv = float(net.bytes_recv / 1024 / 1024)

        # 发送数据
        net_sent = float(net.bytes_sent / 1024 / 1024)

        self.output_file.write("%.2f,%.2f,%.2f,%.2f,%.2f\n" % (
            time_stamp, cpu_per_format, memory_per, net_recv, net_sent
        )
                               )
        self.output_file.flush()


if __name__ == "__main__":
    snoop = AllSnoop()
    snoop.main_loop("data/all.csv", 1)
