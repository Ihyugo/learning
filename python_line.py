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
plt.scatter(X[:50,0], X[:50,1],color='red', marker='o', label='setosa')
#　品種versicolorのプロット(青の☓)
plt.scatter(X[50:100,0], X[50:100,1],color='blue', marker='x', label='versicolor')
#軸のラベル設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

#text p.29
#凡例の設定（左上に配置）
from Perceptron import Perceptron
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

#text p.31
#決定境界のプロット
from iris_marker import *
plot_decision_regions(X, y, classifier=ppn)
#軸のラベル設定
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')
plt.show()

#テキスト　p.37

from Adline import *
#勾配領域を１行２勾配領域を１行２列に分割
fig, ax =  plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
#勾配降下法によるADLINEの学習(学習率　eta=0.01)
ada1 = AdalineGD(n_iter=10, eta=0.01,).fit(X,y)
#エポック数とコストの関係を表す折れ線グラフのプロット（縦軸のコストは常用対数）
ax[0].plot(range(1,len(ada1.cost_)+1), np.log10(ada1.cost_), marker='o')
#軸のラベルの設定
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error')
#タイトルの設定
ax[0].set_title('Adaline - Learning-error')
#勾配降下法によるADLINEの学習（学習率　eta=0.0001)
ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
#エポック数うとコストの関係を表す折れ線グラフのプロット
ax[1].plot(range(1,len(ada2.cost_)+1), ada2.cost_, marker='o')
#軸のラベルの設定
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Sum-squared-error')
#タイトルの設定
ax[1].set_title('Adaline - Learning rate 0.0001')
plt.show()
