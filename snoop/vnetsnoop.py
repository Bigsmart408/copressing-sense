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
import libvirt
from xml.etree import ElementTree


class VnetSnoop:

    def main_loop(self, output_filename, interval):
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
            self.cpurecord(cur_time)

    def cpurecord(self, time_stamp):
        # cpu逻辑核数
        conn = libvirt.open("qemu:///system")  # 打开本地虚拟机信息
        for id in conn.listDomainsID():
            # 获取当前虚拟机id，name以及一些基础info
            domain = conn.lookupByID(id)

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

            self.output_file.write("%.2f,%.2f,%.2f\n" % (
                time_stamp, ifaceinfo[0], ifaceinfo[4]
            )
                                   )
            self.output_file.flush()


if __name__ == "__main__":
    snoop = VnetSnoop()
    snoop.main_loop("vnet.csv", 1)
