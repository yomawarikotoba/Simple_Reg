import numpy as np


class DenseLayer:
    def __init__(
        self, input_size, num_neurons, activation_func, activation_derivative, name=""
    ):
        self.name = name
        # self.weightsに0.1をかけているのは勾配消失を防ぐため
        # biasは0埋めで初期化（サイズはneuron数と一致させる）
        self.weights = np.random.randn(input_size, num_neurons)
        self.biases = np.zeros(
            (1, num_neurons)
        )  # (()) <- このように書くのはちゃんと理由があって()一つだと二つ目の数をdtypeとして認識してしまうからそれを回避するため

        # 活性化関数とその導関数を保持
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative

        # 逆伝播で使う値を保持するための変数を準備
        """なぜこれをNoneで定義しておくのか、空のリストではだめなのか -> だめ。
        1.初期値がリストなのに後から入る値はNumpy配列であり、予期せぬエラーが起こる
        2.今回は計算結果である単一のNumpy配列を代入することになるため、値がまだないという言いを表すNoneのほうが適切"""
        self.last_input = None
        self.last_linear_combination = None

        # 計算された勾配を保存するための変数
        self.weights_gradient = None
        self.biases_gradient = None

    def forward(self, inputs):
        # print(f" -> Forward:層{self.name}が呼ばれた")
        # ここでは順伝播を行い逆伝播で必要な値をインスタンス変数に保存する
        self.last_input = inputs

        # 線形結合
        linear_combination = np.dot(self.last_input, self.weights) + self.biases

        # 逆伝播を行うために線形結合の結果も保存
        self.last_linear_combination = linear_combination

        # 活性化関数を適用
        output = self.activation_func(self.last_linear_combination)

        return output

    def backward(self, upstream_gradient):
        # 教科書の245-246に準拠
        # 逆伝播を行い各パラメータの勾配と前の層に伝える勾配を計算する
        # upstream_gradientは後ろの層から伝わってきた勾配

        # 活性化関数の微分を適用する
        # print(f" <- Backward: 層{self.name}がよばれた　self.last_input is None?->{self.last_input is None}")
        # d_linearとは∂E/∂zのこと
        d_linear = upstream_gradient * self.activation_derivative(
            self.last_linear_combination
        )

        # 各パラメータの勾配を計算する
        """weights_gradientとbiases_gradientをnum_samplesで割るのはバッチサイズ（一度に処理するサンプル数）が
        変わっても学習の安全性を保つため、計算したweights_gradientとbiases_gradientはバッチ内の全サンプルの勾配の合計値であり、
        バッチサイズが10倍になれば合計値も単純に10倍になる。
        勾配の大きさがバッチサイズに依存してしまうと学習率の調整が面倒。
        よってサンプル数で割ってあげることで「サンプル1つ当たりの平均的な勾配」を求められ、
        勾配の大きさがバッチサイズから独立して同じ学習率で安定した学習が期待できる。
        """
        num_samples = self.last_input.shape[0]

        """∂z/∂wが前の層からの入力、コードではself.last_inputにあたる（実際はその転置）
        w(τ+1) = w(τ) - η∇E(w(τ)) の式において∇E(w(τ))を計算しているのがここ
        ∂E/∂w=∂E/∂z * ∂z/∂w として∂E/∂z=d_linear、∂z/∂wがnp.dot()で定義されている"""
        self.weights_gradient = np.dot(self.last_input.T, d_linear) / num_samples

        """biasの場合も連鎖律を考える。
        ∂E/∂b=∂E/∂z * ∂z/∂b として∂z/∂b=1よりバイアスの勾配は∂E/∂b=∂E/∂z
        順伝播ではバイアスベクトル(1, neuron数)がバッチ内のすべてのサンプルにコピーされて足されている
        逆伝播ではその逆の操作、つまり各サンプルについて計算された勾配d_linearをすべて合計して集約する必要がある
        というのも連鎖律ではある変数で全体の誤差Eを微分する場合、その変数が影響を与えているすべての経路からの勾配を合計する必要があるため。
        ∂E/∂b = sum_i=1~m(∂E/∂z_ij)
        よって各ニューロン毎に（axis=0は縦方向を意味しているからこれに該当）すべてのbias勾配を足し合わせて、
        次元の形状をbiasのそれにそろえてあげるという処理をここでしている"""
        self.biases_gradient = np.sum(d_linear, axis=0, keepdims=True) / num_samples

        # 前の層に伝える勾配を計算する
        downstream_gradient = np.dot(d_linear, self.weights.T)

        return downstream_gradient
