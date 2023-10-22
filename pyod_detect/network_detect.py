import time

import numpy as npy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as mfm

from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.feature_bagging import FeatureBagging

class NetDetect():
    def detect(self, begin, end):
        df = pd.read_csv("/home/bfq/project788067-124730-pre/network.csv")
        if(end >= len(df)):
            end = len(df) - 1
        df = df.iloc[begin:(end + 1)]
        ticks = []
        
        for i in range(begin, end + 1):
            ticks.append(time.strftime('%H:%M:%S', time.localtime(df['TICKS'][i])))

        clf = LOF(n_neighbors=20)

        x = npy.array(df['TICKS']).reshape(-1, 1)
        y1 = npy.array(df['RX_KB']).reshape(-1, 1)
        y2 = npy.array(df['TX_KB']).reshape(-1, 1)

        train_data1 = npy.concatenate((x, y1),axis=1)
        train_data2 = npy.concatenate((x, y2),axis=1)

        xx,yy = npy.meshgrid(npy.linspace(0,1,200),npy.linspace(0,1,200))

        clf.fit(train_data1)

        train_pred1 = clf.labels_
        train_scores1 = clf.decision_scores_

        dfx = df
        dfx['outlier'] = train_pred1.tolist()

        IX1 = npy.array(dfx['TICKS'][dfx['outlier']==0]).reshape(-1, 1)
        IX2 = npy.array(dfx['RX_KB'][dfx['outlier']==0]).reshape(-1, 1)

        OX1 = npy.array(dfx['TICKS'][dfx['outlier']==1]).reshape(-1, 1)
        OX2 = npy.array(dfx['RX_KB'][dfx['outlier']==1]).reshape(-1, 1)

        plt.figure(figsize=(13,13))
        plt.scatter(IX1,IX2,c='white',s=20,edgecolor = 'k')
        plt.scatter(OX1,OX2,c='black',s=20,edgecolor = 'k')

        plt.savefig('network_recv_' + (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(df['TICKS'][i]))) + ".png")
        plt.cla()
        clf.fit(train_data2)

        train_pred2 = clf.labels_
        train_scores2 = clf.decision_scores_

        dfx = df
        dfx['outlier'] = train_pred2.tolist()

        IX1 = npy.array(dfx['TICKS'][dfx['outlier']==0]).reshape(-1, 1)
        IX2 = npy.array(dfx['TX_KB'][dfx['outlier']==0]).reshape(-1, 1)

        OX1 = npy.array(dfx['TICKS'][dfx['outlier']==1]).reshape(-1, 1)
        OX2 = npy.array(dfx['TX_KB'][dfx['outlier']==1]).reshape(-1, 1)

        plt.scatter(IX1,IX2,c='white',s=20,edgecolor = 'k')
        plt.scatter(OX1,OX2,c='black',s=20,edgecolor = 'k')

        plt.xticks(dfx['TICKS'], ticks, rotation=40)
        plt.savefig('network_send_' + (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(df['TICKS'][i]))) + ".png")
