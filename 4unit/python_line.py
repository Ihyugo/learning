
#text p.94

import pandas as pd
from io import StringIO
#　サンプルデータを作成
csv_data = '''A,B,C,D
1.0,2.0,3.0,4.0
5.0,6.0,,8.0
10.0,11.0,12.0,'''
# Python2.7を使用している場合は文字列をunicodeに変換する必要がある
#ｃｓｖ_data = unicode(csv_data)
#サンプルデータを読み込む
df = pd.read_csv(StringIO(csv_data))
print(df)
print("\n")
#各特徴量の欠測値をカウント
print(df.isnull().sum())
print("\n")

# 欠測値削除

#欠測値を含む行を削除
print(df.dropna())
print("\n")
#欠測値を含む行を削除
print(df.dropna(axis=1))
#すべての列がNaNである行だけを削除
print(df.dropna(how='all'))
#非NaN値が4つ未満の行を削除
print(df.dropna(thresh=4))
#特定の列（この場合は'C'）にNaNが含まれている行だけ削除
print(df.dropna(subset=['C']))

# 平均値補完 text p.96
print("\n")
from sklearn.preprocessing import Imputer
# 欠測値補完のインスタンスを生成（平均値補完）
imr = Imputer(missing_values='NaN', strategy='mean', axis=1) #axis=0で列の平均、axis=1で行の平均
# データを適合
imr = imr.fit(df)
# 補完を実行
imputed_data = imr.transform(df.values)
print(imputed_data)

# category data の処理
# text p.98
print("\n Tシャツ \n")
import pandas as pd
#サンプルデータを生成（Tシャツの色、サイズ、価格、クラスラベル）
df = pd.DataFrame([
['green', 'M', 10.1, 'class1'],
['red', 'L', 13.5, 'class2'],
['blue', 'XL', 15.3, 'class1']])
#列名を指定
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

# Tシャツのサイズと整数を対応させるディクショナリを生成
size_mapping = {'XL': 3, 'L': 2, 'M': 1}
#　Tシャツのサイズを整数に変換
df['size'] = df['size'].map(size_mapping)
print(df)

import numpy as np
#　クラスラベルと整数を対応させるディクショナリを生成
class_mapping = {label:idx for idx,
    label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)

# クラスラベルを整数に変換
df['classlabel'] = df['classlabel'].map(class_mapping)
print(class_mapping)

#　整数とクラスラベルを対応させるディクショナリを生成
inv_class_mapping = {v: k for k, v in class_mapping.items()}
# 整数からクラスラベルに変換
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

from sklearn.preprocessing import LabelEncoder
# ラベルエンコーダのインスタンスを生成
class_le = LabelEncoder()
#　クラスラベルから整数に変換
y = class_le.fit_transform(df['classlabel'].values)
print(y)

# クラスラベルを文字列に戻す
print(class_le.inverse_transform(y))

# Tシャツの色、サイズ、価格を抽出
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)

# text p.102

from sklearn.preprocessing import OneHotEncoder
#one-hot エンコーダの生成
ohe = OneHotEncoder(categorical_features=[0])
# one-hotエンコーディングの生成
print(ohe.fit_transform(X).toarray())

# one-hot絵コーディングの実行
pd.get_dummies(df[['price', 'color', 'size']])

#wineデータセットを読み込む

df_wine = pd.read_csv('../wine.csv', header=None)
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

from sklearn.model_selection import train_test_split
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
