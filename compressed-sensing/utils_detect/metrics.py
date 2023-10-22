import numpy as np
from sklearn.metrics import confusion_matrix




def dynamic_threshold(score: np.ndarray, ratio: float = 3):
    '''
    动态阈值
    ---
    ### Parameters:
        score: np.ndarray 异常评分
        ratio: float (default = 3) 标准差比例，例：threshold = mean + `ratio` * std
    ### Returns:
        proba: 预测异常
        (Temporary Delete)// threshold: 对于该段输入的score 选择的阈值
    '''
    std = np.std(score)
    mean = np.mean(score)
    threshold = mean + ratio * std
    if threshold >= 1:
        threshold = 0.999
    elif threshold <= 0:
        threshold = 0.001
    proba = np.array(score>threshold, dtype=np.int)
    proba = np.array(proba, dtype=int)
    return proba


def sliding_anomaly_predict(score: np.ndarray, window_size: int = 1440, stride: int = 10, ratio: float = 3):
    '''
    滑动窗口的动态阈值异常预测
    ---
    ### Parameters:
        score: np.ndarray 异常评分
        window_size: int (default = 1440) 滑动窗口大小
        stride: int (default = 10) 滑动窗口步长
        ratio: float (default = 3) 标准差比例，例：threshold = mean + `ratio` * std
    ### Return:
        predict: np.ndarray 根据滑动窗口的动态阈值对异常做出 0、1预测
    '''
    predict = np.zeros(score.shape, dtype=int)
    ny = score.shape[0]
    start = 0
    while start < ny:
        predict[start:start +
                window_size] = dynamic_threshold(score[start:start+window_size], ratio)
        start += stride
    return predict




