import onnxruntime as ort
import numpy as np
import time

from dataset import DetectDataSet
import pyod_detect
from pyod_detect import mem_detect,cpu_detect,network_detect
from pyod_detect.cpu_detect import CpuDetect
from pyod_detect.mem_detect import MemDetect
from pyod_detect.network_detect import NetDetect


class LstmDetect():

    def detect(self):
        mem_detect = MemDetect()
        cpu_detect = CpuDetect()
        net_detect = NetDetect()

        ort_sess = ort.InferenceSession("model.onnx")

        dataset_cpu = DetectDataSet("cpu.csv")
        dataset_mem = DetectDataSet("mem.csv")
        dataset_net = DetectDataSet("network.csv")

        for idx in range(len(dataset_cpu)):
            input, label, timestamp = dataset_cpu[idx]
            input = np.expand_dims(input, axis=1)
            input = np.expand_dims(input, axis=0)

            outputs = ort_sess.run(None, {'input': input})
            loss = np.sum(np.abs(outputs[0] - label))
            if (loss > 70):
                cpu_detect.detect(idx - 20, idx + 20)

        for idx in range(len(dataset_mem)):
            input, label, timestamp = dataset_mem[idx]
            input = np.expand_dims(input, axis=1)
            input = np.expand_dims(input, axis=0)

            outputs = ort_sess.run(None, {'input': input})
            loss = np.sum(np.abs(outputs[0] - label))
            if (loss > 100):
                mem_detect.detect(idx - 20, idx + 20)

        for idx in range(len(dataset_net)):
            input, label, timestamp = dataset_net[idx]
            input = np.expand_dims(input, axis=1)
            input = np.expand_dims(input, axis=0)

            outputs = ort_sess.run(None, {'input': input})
            loss = np.sum(np.abs(outputs[0] - label))
            if (loss > 50000):
                net_detect.detect(idx - 20, idx + 20)
