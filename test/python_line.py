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

X_std = np.copy(X)
#各列の標準化
X_std[:,0] = (X[:,0] - X[:,0].mean()) / X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean()) / X[:,1].std()

from iris_marker import *
from test import *
#勾配降下法によるADLINEの学習（標準化後、学習率 eta=0.01)

from ada2 import AdlineSGD
#確率的勾配降下法によるADALINEの学習
ada = AdlineSGD(n_iter=30, eta=0.1, random_state=1)
#モデルへの適合
ada.fit(X_std, y)
# 境界領域のプロット
plot_decision_regions(X_std, y, classifier=ada)
#タイトルの設定
plt.title('Adline - Stochastic Gradient Descent')
#軸のラベル設定
plt.xlabel('sepal length [standard]')
plt.ylabel('petal length [standard]')
#　凡例の設定
plt.legend(loc='upper left')
#　図の表示
plt.show()

#エポックとコストの折れ線グラフのプロット
plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
# 軸のラベルの設定
plt.xlabel('Epochs')
plt.ylabel('Average COst')
#　図の表示
plt.show()
