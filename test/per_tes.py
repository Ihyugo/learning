# text p.28
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
                'machine-learning-databases/iris/iris.data', header=None)
df.tail()
import matplotlib.pyplot as plt
import numpy as np
#1-100行目の目的て変数の抽出
y = df.iloc[0:100, 4].values
# Iris-setosaを-1、Iris-virginicaを1に変換
y = np.where(y == 'Iris-setosa', -1, 1)
#1-100行目の1,3列目の抽出
X = df.iloc[0:100, [0, 2]].values
#　品種setosaのプロット(赤の○)
# plt.scatter(X[:50,0], X[:50,1],color='red', marker='o', label='setosa')
# #　品種versicolorのプロット(青の☓)
# plt.scatter(X[50:100,0], X[50:100,1],color='blue', marker='x', label='versicolor')
# #軸のラベル設定
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
# plt.show()

#text p.29
#凡例の設定（左上に配置）
from test import Perceptron
#パーセプロトンのオブジェクトの作製（インスタンス化）
ppn = Perceptron(eta=0.1, n_iter=10)
#　トレーニングデータへのモデル適合
ppn.fit(X,y)
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
#　軸のラベル設定
plt.xlabel('sepal length [cm]')
plt.ylabel('Numbet of misclassifications')
# 図の表示
plt.legend(loc='upper left')
#図の表示
plt.show()
