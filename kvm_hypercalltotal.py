
from __future__ import print_function
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


# header
print("%-18s %-16s %-6s %s" % ("TIME(s)", "COMM", "PID", "EVENT"))

# format output
while 1:
    try:
        (task, pid, cpu, flags, ts, msg) = b.trace_fields()
    except ValueError:
        continue
    print("%-18.9f %-16s %-6d %s" % (ts, task, pid, msg))