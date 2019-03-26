# text p.102

import pandas as pd
import numpy as np
#wineデータセットを読み込む

df_wine = pd.read_csv(
'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data',
 header=None)
print(df_wine)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                   'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Rue',
                   'OD280/OD315 of diluted wines', 'Proline']
# クラスラベルを表示
print('Class labels', np.unique(df_wine['Class label']))
# wineデータセットの戦闘5行を表示
df_wine.head()

# text p.104

from sklearn.cross_validation import train_test_split
# 特徴量とクラスラベルを別々に抽出
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
# トレーニングデータとテストデータに分割
# 全体の30%をテストデータにする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.preprocessing import MinMaxScaler
# min-maxスケーリングのインスタンスを生成
mms = MinMaxScaler()
# トレーニングデータをスケーリング
X_train_norm = mms.fit_transform(X_train)
# テストデータをスケーリング
X_test_norm = mms.transform(X_test)


# text p.106

from sklearn.preprocessing import StandardScaler
# 標準化のインスタンスを生成（平均=0,　標準偏差=1に変換）
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)

# text p.110  l1正則
from sklearn.linear_model import LogisticRegression
# L1正則化閉じスティック回帰のインスタンスを生成(逆正則化パラメータ C=0.1)
lr = LogisticRegression(penalty='l1', C=0.1)
#　トレーニングデータに適合
lr.fit(X_train_std, y_train)
#　トレーニングデータに対する正解率の表示
print('Training accuracy:', lr.score(X_train_std, y_train))
# テストデータに対する正解率の表示
print('Test accuracy:', lr.score(X_train_std, y_train))

# 切片の表示
print(lr.intercept_)
#　重み係数の表示
print(lr.coef_)

import  matplotlib.pyplot as plt
#　描画の準備
fig = plt.figure()
ax = plt.subplot(111)
# 各係数の色のリスト
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'gray', 'indigo', 'orange']
# からのリストを生成（重み係数、　逆正則化パラメータ）
weights, params = [], []
# 空のリストを生成（重み係数、逆正則化パラメータ）

for c in np.arange(-4, 6):
    coefficient = 10.0**c
    lr = LogisticRegression(penalty='l1', C=coefficient, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(coefficient)

# 重み係数をnumpy配列に変換
weights = np.array(weights)
# 各重み係数をプロット
for column, color in zip(range(weights.shape[1]), colors):
    #　横軸を逆正則化パラメータ、、縦軸を重み係数とした折れ線グラフ
    plt.plot(params, weights[:, column], label=df_wine.columns[column+1], color=color)

# y=0に黒い破線を引く
plt.axhline(0, color='black', linestyle='--', linewidth=3)
#横軸の範囲を設定
plt.xlim([10**(-5), 10**5])
#　軸のラベルの設定
plt.xlabel('C')
plt.ylabel('weight coefficient')
plt.xscale('log')
# 凡例の設定
plt.legend(loc = 'upper left')
ax.legend(loc='upper center', bbox_to_anchor=(1.38, 1.03), ncol=1, fancybox=True)
# 図の表示
plt.show()

# text p.117

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from SBS import *
# k近傍分類器のインスタンスを生成（近傍点数=2）
knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
#　逐次交代選択を実行
sbs.fit(X_train_std, y_train)

# 近傍点の個数のリスト（13, 12, ..... ,1)
k_feat = [len(k) for k in sbs.subsets_]
# 横軸を近傍点の個数、縦軸をスコアとした折れ線グラフのプロット
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

# text p.113

k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])
#13個すべての特徴量を用いてモデルに適合
knn.fit(X_train_std, y_train)
# トレーニングの正解率を出力
print('Training accuracy:', knn.score(X_train_std, y_train))
# テストの正解率を出力
print('Test accuracy:', knn.score(X_test_std, y_test))

# 5このの特徴量を用いてモデルに適合
knn.fit(X_train_std[:, k5], y_train)
# トレーニングデータの正解率を出力
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
# テストの正解率を出力
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))


from sklearn.ensemble import RandomForestClassifier
# wineデータセットの特徴量の名称
feat_labels = df_wine.columns[1:]
# ランダムフォレストオブジェクトの生成
# （木の個数=10000、　すべてのコアを用いて並列計算を実行)
forest = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)
# モデルに適合
forest.fit(X_train, y_train)
# 特徴量の重要度を抽出
importances = forest.feature_importances_
# 重要度の降順で特徴量のインデックスを抽出
indices = np.argsort(importances)[::-1]
# 重要度の講ずんで特徴量の名称、重要度を表示
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" %
                (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importances')
plt.bar(range(X_train.shape[1]), importances[indices], color='lightblue', align='center')
plt.xticks(range(X_train.shape[1]), feat_labels[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.tight_layout()
plt.show()

# 重量度が0.15以上の特徴量を抽出
X_selected = forest.transform(X_train, threshold=0.15)
print(X_selected.shape)
