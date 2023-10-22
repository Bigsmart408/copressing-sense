#! /bin/python3
import multiprocessing as mp
from time import sleep, strftime, time
import argparse
from yaml import parse
from cpusnoop import CpuSnoop
from memsnoop import MemSnoop
from netsnoop import NetSnoop
from allsnoop import  AllSnoop
from vcpusnoop import VcpuSnoop
from vnetsnoop import VnetSnoop
from vmemsnoop import VmemSnoop
import subprocess


class TOPSnoop():
    def __init__(self) -> None:
        """初始化类，并调用参数处理函数
        """
        self.cpu_snoop = CpuSnoop()
        self.mem_snoop = MemSnoop()
        self.network_snoop = NetSnoop()
        self.vcpu_snoop = VcpuSnoop()
        self.vmem_snoop = VmemSnoop()
        self.vnetwork_snoop = VnetSnoop()
        self.all_snoop = AllSnoop()

    def run(self):
        """对外接口，启动监控进程
        使用子进程的方式来实现对多种数据同时进行监控
        """
        
        cpu_snoop_process = mp.Process(target=self.cpu_snoop.main_loop, args=("csv/cpu.csv", 1))
        mem_snoop_process = mp.Process(target=self.mem_snoop.main_loop, args=("csv/mem.csv", 1))
        net_snoop_process = mp.Process(target=self.network_snoop.mainloop, args=("csv/net.csv", 1))
        all_snoop_process = mp.Process(target=self.all_snoop.main_loop,args=("csv/all.csv",1))
        vcpu_snoop_process = mp.Process(target=self.vcpu_snoop.main_loop, args=("csv/vcpu.csv", 1))
        vmem_snoop_process = mp.Process(target=self.vmem_snoop.main_loop, args=("csv/vmem.csv", 1))
        vnet_snoop_process = mp.Process(target=self.vnetwork_snoop.main_loop, args=("csv/vnet.csv", 1))
        cpu_snoop_process.start()
        mem_snoop_process.start()
        net_snoop_process.start()
        vcpu_snoop_process.start()
        vmem_snoop_process.start()
        vnet_snoop_process.start()


if __name__ == "__main__":
    top_snoop = TOPSnoop()
    top_snoop.run()




