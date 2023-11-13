from __future__ import print_function
import os
import sys
import time
import argparse
import multiprocessing as mp
import subprocess
sys.path.append(os.getcwd() + '/snoop')
sys.path.append(os.getcwd() + '/compressed-sensing')

import runsnoop 
import detect
from recorder import record
from bcc import BPF

b = BPF(text="""
BPF_HASH(start, u8, u8);

TRACEPOINT_PROBE(kvm,kvm_hypercall){
        bpf_trace_printk("HYPERCALL nr : %d\\n",args->nr);
        bpf_trace_printk("HYPERCALL a0 : %d\\n",args->a0);
        bpf_trace_printk("HYPERCALL a1 : %d\\n",args->a1);
        bpf_trace_printk("HYPERCALL a2 : %ld\\n",args->a2);
        bpf_trace_printk("HYPERCALL a3 : %ld\\n",args->a3);
};
""")
if __name__ == '__main__':
    top = runsnoop.TOPSnoop()
    top_process = mp.Process(target=top.run, args=()) 
    top_process.start()
    last = [0, 0, 0]
    time.sleep(50)
    while True:
        try:
            (task, pid, cpu, flags, ts, msg) = b.trace_fields()
        except ValueError:
            continue
        a = len(open("csv/data/vall.csv", 'r').readlines())
        detect.detect("csv/data/vall.csv", "outcome/csv/vall.csv", a-30)
        
