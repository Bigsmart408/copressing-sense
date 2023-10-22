import time

import numpy as npy
import pandas as pd
import matplotlib.pyplot as plt

from pyod.models.lof import LOF
from pyod.models.knn import KNN

class MemDetect():

    def detect(self, begin, end):
        df = pd.read_csv("/home/bfq/project788067-124730-pre/mem.csv") 
        if(end >= len(df)):
            end = len(df) - 1
        df = df.iloc[begin:(end + 1)]
        ticks = []

        for i in range(begin, end + 1):
            ticks.append(time.strftime('%H:%M:%S', time.localtime(df['TICKS'][i])))

        clf = LOF(n_neighbors=20)

        x = npy.array(df['TICKS']).reshape(-1, 1)
        y = npy.array(df['size(B)']).reshape(-1, 1)
        train_data = npy.concatenate((x, y),axis=1)
        xx,yy = npy.meshgrid(npy.linspace(0,1,200),npy.linspace(0,1,200))

        clf.fit(train_data)

        train_pred = clf.labels_
        train_scores = clf.decision_scores_

        dfx = df
        dfx['outlier'] = train_pred.tolist()

        IX1 = npy.array(dfx['TICKS'][dfx['outlier']==0]).reshape(-1, 1)
        IX2 = npy.array(dfx['size(B)'][dfx['outlier']==0]).reshape(-1, 1)

        OX1 = npy.array(dfx['TICKS'][dfx['outlier']==1]).reshape(-1, 1)
        OX2 = npy.array(dfx['size(B)'][dfx['outlier']==1]).reshape(-1, 1)
        plt.figure(figsize=(13,13))
        plt.scatter(IX1,IX2,c='white',s=20,edgecolor = 'k')
        plt.scatter(OX1,OX2,c='black',s=20,edgecolor = 'k')
        plt.xticks(dfx['TICKS'], ticks, rotation=40)

        plt.savefig('mem_' + (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(df['TICKS'][i]))) + ".png")

