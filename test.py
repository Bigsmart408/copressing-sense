import datetime
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
from bcc import BPF

# load BPF program
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
        
        detect.detect("snoop/data/all.csv", "outcome/csv/all.csv", 0)