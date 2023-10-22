<img src = /img/title.png/>

# LMY

我们的队伍来自**哈尔滨工业大学（深圳）**，基本情况如下：

|赛题|<a href=https://github.com/oscomp/porj120-anomaly-detection-for-os>porj120-anomaly-detection-for-os</a>|
|-|-|
|小组成员|满洋、于伯淳、李怡凯|
|指导老师|夏文、李诗逸|

# 赛题完成情况

项目的目标如下：

1. 实现进程级占用资源（CPU占用率、内存占用、网络流量）和系统调用数据的采集；
2. 对采集得到的数据进行处理，从中检测到操作系统可能发生的异常（CPU冲高、内存泄漏、网络流量异常等）；
3. 适配不同的实际场景和运行环境，提高项目的可移植性

当前完成进度如下：

|目标编号|完成情况|说明|
|:-:|:-:|:-:|
|1|基本完成(约90%)|能够以csv格式，以时序记录目标进程的CPU占用、内存占用、网络流量、系统调用四种关键数据|
|2|大致完成(约80%)|以lstm神经网络为主，辅以knn等算法，在合理的阈值设定下能对明显异常达到100%的准确率，并可以灵活调整阈值以调整对异常灵敏度|
|3|部分完成(约50%)|项目主体采用Python的bcc框架、pyod和pytorch等工具，移植性的瓶颈主要是ebpf在不同内核版本的差异|
|总计|大致完成(约80%)|已经对cpu冲高、内存占用异常、网络流量异常实现了一套解决方案，后续将考虑算法的多样性以及项目在不同平台的移植性问题。目前项目内存和cpu占用开销较大，约为300MB；异常检测速度较快，每条数据的检测时间平均能达到0.01s；准确率较为理想，对手动注入的异常能达到100%的检出，对选取的开源数据集中的异常也都实现了检出。|

# 项目代码结构

```
|-- main.py                 # 入口
|-- model.onnx              # LSTM 模型
|-- requirements.txt        # Python 依赖
|-- snoop                   # 监控模块
|   |-- top_snoop.py
|   |-- cpu_snoop.py
|   |-- mem_snoop.py
|   |-- network_snoop.py
|   |-- syscall_snoop.py
|   |-- utils.py
|   |-- README.md
|
|-- lstm_detect             # LSTM 检测模块
|   |-- dataset.py
|   |-- dataset_torch.py
|   |-- detect.py
|   |-- detect_onnx.py      # 检测入口
|   |-- input_data.toml
|   |-- modeling.py
|   |-- requirements.txt
|   |-- README.md
|
|-- pyod_detect             # PyOD 检测模块和绘图模块
|   |-- cpu_detect.py
|   |-- mem_detect.py
|   |-- network_detect.py
|
|-- README.md

```


# 文档信息

+ <a href="https://gitlab.eduxiji.net/CH3CHOHCH3/project788067-124730/-/blob/master/%E6%BC%94%E7%A4%BA%E8%A7%86%E9%A2%91.mp4">项目演示视频</a>
+ <a href="https://gitlab.eduxiji.net/CH3CHOHCH3/project788067-124730/-/blob/master/%E9%A1%B9%E7%9B%AE%E6%96%87%E6%A1%A3.pdf">项目文档</a>
