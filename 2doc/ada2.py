from numpy.random import seed
import numpy as np

class AdlineSGD(object):
    """ADAptive LInear NEuron 分類器

    パラメータ
    -----------
    eta : float
        学習率 (0.0より大きく1.0以下の値)
    n_iter : int
        トレーニングデータのトレーニング回数
    属性
    -----------
    w_ : 1次元配列
        適合後の重み
    errors_ : リスト
        各エポックでの誤分類数
    shuffle : bool (デフォルト : True)
        循環を回避するために各エポックでトレーニングデータをシャッフル
    random_state : int (デフォルト : None)
        シャッフルに使用するランダムステートを設定し、重みを初期化

    """

    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        #学習率の初期化
        self.eta = eta
        #トレーニング回数の初期化
        self.n_iter = n_iter
        #　重みの初期化フラグはFlaseに設定
        self.w_initialized = False
        #各エポックでトレーニングデータをシャッフルするかどうかのフラグ
        # を初期化
        self.shuffle = shuffle
        # 引数 random_state　が指定された場合は乱数種を設定
        if random_state:
            seed(random_state)

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
        #重みベクトルの生成
        self._initialize_weights(X.shape[1])
        #コストを格納するリストの作製
        self.cost_ = []
        #トレーニング回数分トレーニングデータを反映
        for i in range(self.n_iter):
            # が指定された場合はトレーニングデータをシャッフル
            if self.shuffle:
                X, y = self._shuffle(X, y)
            #　各サンプルのコストを格納するリストの生成
            cost = []
            #各サンプルに対する計算
            for xi, target in zip(X,y):
                # 特徴量xi と目的変数yを用いた重みの更新とコストの計算
                cost.append(self._update_weights(xi, target))
            # サンプルの平均コストの計算
            avg_cost = sum(cost) / len(y)
            #平均コストを格納
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """重みを最初期化する小音なくトレーニングデータに適合される"""
        # 初期化されていない場合は初期化を実行
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        # 目的変数yの要素数が２以上の場合は
        #  各サンプルの特徴量xiと目的変数targetで重みを更新
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        #目的変数yの要素数が１の場合は
        # サンプル全体の特徴量Xと目的変数yで重みを更新
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """トレーニングデータをシャッフル"""
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """重みを０に初期化"""
        self.w_ = np.zeros(1 + m)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        """ADALINEの学習規則を用いて重みを更新"""
        #活性化関数の出力の計算
        output = self.net_input(xi)
        # 誤差の計算
        error = (target -output)
        #重み　w1, ,,,, wnの更新
        self.w_[1:] += self.eta * xi.dot(error)
        #　重み w0の更新
        self.w_[0] += self.eta * error
        # コストの計算
        cost = 0.5 * error**2
        return cost

    def net_input(self, X):
        """活性化関数の出力を計算"""
        return np.dot(X, self.w_[1:] + self.w_[0])

    def activation(self, X):
        """線形活性化関数の出力を計算"""
        return self.net_input(X)

    def predict(self, X):
        """１ステップ後のクラスラベルを返す"""
        return np.where(self.activation(X) >= 0.0, 1, -1)
