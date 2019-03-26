# text p.49

from sklearn import datasets
import numpy as np
import matplotlib.pyplot  as plt
#　Irisデータ・セットをロード
iris = datasets.load_iris()
# 3,4行目の特徴量を抽出
X = iris.data[:, [2, 3]]
# クラスラベルを取得
y = iris.target
print("Class labels:", np.unique(y))
from sklearn.model_selection import train_test_split
# from sklearn.model_selection import train_test_split
#トレーニングデータとテストデータに分割
# 全体の３０％をテストデータにする
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# text p.50

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# トレーニングデータの平均と標準偏差の計算
sc.fit(X_train)
# 平均と標準偏差を用いて標準化
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron
# エポック数4.0、学習率0.1でパーセプトロンのインスタンスを生成
ppn = Perceptron(max_iter=60, eta0=0.01, random_state=0, shuffle=True)
# トレーニングデータをモデルに適合させる
ppn.fit(X_train_std, y_train)
#　テストデータで予測を実施
y_pred = ppn.predict(X_test_std)
# 誤分類のサンプルの個数を表示
print('Misclassified samples: %d' % (y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
# 分類の正解率を表示
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# text p.52

from irismarker import plot_decision_regions
# トレーニングデータとテストデータの特徴量を行方向に結合
X_combined_std = np.vstack((X_train_std, X_test_std))
#　トレーニングデータとテストデータのクラスラベルを結合
y_combined = np.hstack((y_train, y_test))
# 決定境界のプロット
plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,test_idx=range(105,150))
#　軸のラベルの設定
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
# 凡例の設定
plt.legend(loc = 'upper left')
# 図の表示
plt.show()

# text p.60

from sklearn.linear_model import LogisticRegression
#　ロジスティック回帰のインスタンスを生成
lr = LogisticRegression(C=1000.0, random_state=0)
#　トレーニングデータをモデルに適合させる
lr.fit(X_train_std, y_train)
#　決定境界をプロット
plot_decision_regions(X_combined_std,y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length [standardized')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# text p.65

#　空のリストを生成（重み係数、　逆正則化パラメータ）
weights, params = [], []
# １０個の逆正則化パラメータに対応するロジスティック回帰モデルをそれぞれ処理
for c in np.arange(-5, 5):
    update = 10.**c
    lr = LogisticRegression(C=update, random_state=0)
    lr.fit(X_train_std, y_train)
    weights.append(lr.coef_[1])
    params.append(update)

weights = np.array(weights)
plt.plot(params, weights[:, 0], label='petal length')
plt.plot(params, weights[:, 1], linestyle='-', label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('c')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()

# text p.69

from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std,y_combined,classifier=svm,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
print('Accuracy: %.2f' % svm.score(X_test_std,y_test))
plt.show()

# text p.71-72

np.random.seed(0)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, -1)
print("-------------------------")

print(X_xor)
print(y_xor)
print("-------------------------")
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='b', marker='x', label='1')
plt.scatter(X_xor[y_xor == -1, 0], X_xor[y_xor == -1, 1],
            c='r', marker='s', label='-1')
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.legend(loc='best')
plt.show()

# text p.74

svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10.0)
svm.fit(X_xor,y_xor)
plot_decision_regions(X_xor,y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

# text p.75

svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
svm.fit(X_train_std, y_train)
plot_decision_regions(X_combined_std, y_combined, classifier=svm, test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# text p.76

# RBF カーネルによるSVMのインスタンスを生成（y パラメータを変更）
svm = SVC(kernel='rbf', random_state=0, gamma=10.0, C=1.0)
svm.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined, classifier=svm,
test_idx=range(105,150))

plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()

# text p.83

import matplotlib.pyplot as plt
import numpy as np
#ジニ不順度の関数を定義
def gini(p):
    return (p)*(1 - (p)) + (1 - p)*(1 - (1 - p))

#　エントロピーの関数を定義
def entropy(p):
    return - p*np.log2(p) - (1 - p)*np.log2((1 - p))

#分類誤差の関数を定義
def error(p):
    return 1 - np.max([p, 1 - p])

#　確率を表す配列を生成（0から0.99まで0.01刻み)
x = np.arange(0.0, 1.0, 0.01)
#　配列の値をもとにエントロピー、分類誤差を計算
ent = [entropy(p) if p != 0 else None for p in x]
sc_ent = [e*0.5 if e else None for e in ent]
err = [error(i) for i in x]
#　図の作製を開始
fig = plt.figure()
ax = plt.subplot(111)
#　エントロピー(2種)、ジニ不純度、分類誤差のそれぞれをループ処理
for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                          ['Entropy', 'Entropy (scaled)',
                           'Gini Inpurity', 'Misclassification Error'],
                          ['-', '-', '--', '-.'],
                          ['black', 'lightgray', 'red', 'green', 'cyan']):
    line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)

#  凡例の設定（中央の上に配置）
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
          ncol=3, fancybox=True, shadow=False)
#　2本の水平の破線を引く
ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
#　横軸の上限/下限の設定
plt.ylim([0, 1.1])
plt.xlabel('p(i=1)')
plt.xlabel('Impurity Index')
plt.show()

# text p.84

from sklearn.tree import DecisionTreeClassifier
#　エントロピーを指標とする決定木のインスタンスを生成
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
#決定木のモデルにトレーニングデータを適合させる
tree.fit(X_train, y_train)
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))
plot_decision_regions(X_combined, y_combined, classifier=tree,
                      test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

# make odt file
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file='tree.dot',
                feature_names=['petal length', 'petal width'])

# text p.88

from sklearn.ensemble import RandomForestClassifier
# エントロピーを指標とするランダムフォレストのインスタンスを生成
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10, random_state=1, n_jobs=2)
#　ランダムフォレストのモデルにトレーニングデータを適合させる
forest.fit(X_train, y_train)
plot_decision_regions(X_combined, y_combined, classifier=forest,
                      test_idx=range(105, 150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()
print("Accuracy: %.2f" % forest.score(X_train, y_train))

# text p.90

from sklearn.neighbors import KNeighborsClassifier
# k近傍法のインスタンスを生成
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train) # k近傍法のモデルにトレーニングデータを適合させる
plot_decision_regions(X_combined_std, y_combined, classifier=knn,
                      test_idx=range(105, 150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.show()
print("Accuracy: %.2f" % knn.score(X_train_std, y_train))
