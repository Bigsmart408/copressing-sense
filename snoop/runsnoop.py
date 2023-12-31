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
from vallsnoop import Vallsnoop


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
        self.vall_snoop = Vallsnoop()

    def run(self):
        """对外接口，启动监控进程
        使用子进程的方式来实现对多种数据同时进行监控
        """
        
        """cpu_snoop_process = mp.Process(target=self.cpu_snoop.main_loop, args=("csv/data/cpu.csv", 0.5))
        mem_snoop_process = mp.Process(target=self.mem_snoop.main_loop, args=("csv/data/mem.csv", 0.5))
        net_snoop_process = mp.Process(target=self.network_snoop.mainloop, args=("csv/data/net.csv", 0.5))
        vcpu_snoop_process = mp.Process(target=self.vcpu_snoop.main_loop, args=("csv/data/vcpu.csv", 0.5))
        vmem_snoop_process = mp.Process(target=self.vmem_snoop.main_loop, args=("csv/data/vmem.csv", 0.5))
        vnet_snoop_process = mp.Process(target=self.vnetwork_snoop.main_loop, args=("csv/data/vnet.csv", 0.5)) """
        all_snoop_process = mp.Process(target=self.all_snoop.main_loop,args=("csv/data/all.csv",0.5))
        vall_snoop_process = mp.Process(target=self.vall_snoop.main_loop,args=("csv/data/vall.csv",0.5))
        """cpu_snoop_process.start()
        mem_snoop_process.start()
        net_snoop_process.start()
        vcpu_snoop_process.start()
        vmem_snoop_process.start()
        vnet_snoop_process.start()"""
        all_snoop_process.start()
        vall_snoop_process.start()



if __name__ == "__main__":
    top_snoop = TOPSnoop()
    top_snoop.run()




