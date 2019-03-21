import numpy as np
class Perceptron(object):
    print(object)
    """パーセプトロンの分類器

    パラメータ
    ----------
    eta : float
        学習率 (0.0より大きく1.0以下の値)
    n_iter : int
        トレーニングデータのトレーニング回数
    属性
    ----------
    w_ : 1次元配列
        適合後の重み
    errors_ : リスト
        各エポックでの誤分類数
    """
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        """トレーニングデータに適合させる

        パラメータ
        ---------
        X : {配列のようなデータ構造},shape = {n_samples, n_features}
            トレーニングデータ
            n_sampleはサンプルの個数、n_featureは特徴量の個数
        Y : 配列のようなデータ構造、 shape = {n_samples}
            目的変数

            戻り値
        ---------
        self: object

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter): # トレーニング回数分トレーニングデータを反復
            errors = 0
            for xi, target in zip(X, y):#各サンプルで重みを更新
                #重み　w1,,,,wmの更新
                #Δwj = η(yi - y'i)xji (j = 1,,,,m)
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                #　重みw0の更新：Δw- = η(yi - y'i)
                self.w_[0] += update
                #重みの更新が0でない場合は誤分類としてカウント
                errors += int(update != 0.0)
            #反復回数ごとの誤差を格納
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """総入力を計算"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """1ステップ後のクラスラベルを返す"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
