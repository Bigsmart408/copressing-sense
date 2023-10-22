import time

import numpy as npy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.feature_bagging import FeatureBagging
from cpu_detect import CpuDetect
from mem_detect import MemDetect
from network_detect import NetDetect

cpu_detect_obj = CpuDetect()
mem_detect_obj = MemDetect()
network_detect_obj = NetDetect()



begin = 0
end  = 14
detect(0,1000)
