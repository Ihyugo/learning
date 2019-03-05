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
from iris_marker import *

from test import *
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

# text p.39
#データのコピー
X_std = np.copy(X)
#各列の標準化
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

#勾配降下法によるADLINEの学習（標準化後、学習率 eta=0.01)
ada = AdalineGD(n_iter=15, eta=0.01)
# モデルの適合
ada.fit(X_std, y)
#　境界領域のプロット
plot_decision_regions(X_std, y, classifier=ada)
#　タイトルの設定
plt.title('Adaline - Grandient Descent')
#　軸のラベルの設定
plt.xlabel('sepal length [standard]')
plt.ylabel('petal length [standard]')
#　凡例の設定
plt.legend(loc='upper left')
#　図の表示
plt.show()

#　エポック数とコストの関係を表す折れ線グラフのプロット
plt.plot(range(1,len(ada.cost_)+1), ada.cost_, marker='o')
# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Sum-squared-error')
#　図の表示
plt.show()

# text p.44
