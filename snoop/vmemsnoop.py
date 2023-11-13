from __future__ import print_function
import psutil
from ctypes import c_uint
import os
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


class VmemSnoop:

    def main_loop(self, output_filename, interval):
        self.output_file = open(output_filename, 'w')
        self.output_file.write("%s,%s\n"
                               % ("TICKS", "Memory_per"))
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
            # 获取当前虚拟机id，name以及一些基础info
            domain = conn.lookupByID(id)

            # 获取当前虚拟机的cpu使用率
            # 周期差:cputime_diff = (cpuTimeNow - cpuTimet seconds ago)
            # 计算实际使用率：%cpu = 100 * cpu_time_diff / (t * nr_cores *109)
            t1 = time.time()  # 可以使用time.time_ns()将数据精确到ns级别
            c1 = int(domain.info()[4])
            time.sleep(1)  # 时间间隔尽量小
            t2 = time.time()
            c2 = int(domain.info()[4])
            c_nums = int(domain.info()[3])
            cpu_per = (c2 - c1) * 100 / ((t2 - t1) * c_nums * 1e9)

            # 获得内存信息,需要添加一个如果unused仍能继续输出
            domain.setMemoryStatsPeriod(10)
            meminfo = domain.memoryStats()
            # 使用try-except进行前期输出
            free_mem = float(meminfo['unused'])
            total_mem = float(meminfo['available'])
            mem_per = ((total_mem - free_mem) / total_mem) * 100
        

                # 获取虚拟机网卡流量信息
            tree = ElementTree.fromstring(domain.XMLDesc())
            ifaces = tree.findall('devices/interface/target')
            for i in ifaces:
                iface = i.get('dev')
                ifaceinfo = domain.interfaceStats(iface)

            # 磁盘信息
            devices = tree.findall('devices/disk/target')
            for d in devices:
                device = d.get('dev')
                try:
                    devinfo = domain.blockInfo(device)
                except libvirt.libvirtError:
                    pass

            self.output_file.write("%.2f,%.2f\n" % (
                time_stamp, mem_per
            )
                                   )
            self.output_file.flush()


if __name__ == "__main__":
    snoop = VmemSnoop()
    snoop.main_loop("vmem.csv", 1)
