import numpy as np
import matplotlib.pyplot as plt
from Layer import DenseLayer
from MSE import MSELoss
from Network import Network

np.random.seed(42)

if __name__ == "__main__":
    # 上のコードはこのpythonファイルがコマンドラインなどから直接実行された場合にのみこのifブロックの中のコードを実行してくださいというおまじない
    # データの準備
    """reshape(-1, 1)によって[-10, -9,…, 10]が列数を1に固定しつつ、
    行数については元の配列の要素数を保つように自動で計算してねという意味の指定が行われ、
    xが21行1列の配列に調整される。"""
    x = np.arange(-10, 11, 1).reshape(-1, 1)
    y_true = x**2

    # 活性化関数とその導関数を定義
    tanh = np.tanh

    # λ式記述法を使っていたがdef記法に修正
    def tanh_derivative(z):
        return 1 - np.tanh(z) ** 2

    def identity(x):
        return x

    def identity_derivative(z):
        return np.ones_like(z)  # 恒等関数の微分は1

    # ネットワークの構築
    net = Network()
    # 隠れ層についての記述（ニューロン数を3にする）
    net.add(
        DenseLayer(
            input_size=1,
            num_neurons=4,
            activation_func=tanh,
            activation_derivative=tanh_derivative,
            name="Hidden Layer",
        )
    )
    # 出力層についての記述
    net.add(
        DenseLayer(
            input_size=4,
            num_neurons=1,
            activation_func=identity,
            activation_derivative=identity_derivative,
            name="Output Layer",
        )
    )

    # 誤差関数とハイパーパラメータ(学習率、訓練回数)の設定
    loss_func = MSELoss()

    learning_rate = 0.0015
    epochs = 150000
    Loss_view_freq = epochs / 1500

    x_epochs = []
    y_loss = []

    # 訓練ループ
    print("訓練を開始します")
    for i in range(epochs):
        # (1)順伝播
        y_pred = net.forward(x)

        # (2)誤差関数の計算
        loss = loss_func.forward(y_pred, y_true)

        # (3)逆伝播
        # まず損失の勾配を計算
        grad = loss_func.backward(y_pred, y_true)
        # ネットワーク全体を逆伝播
        net.backward(grad)

        # (4)パラメータの更新
        net.update(learning_rate)

        # N_epochsごとに損失を表示
        if (i + 1) % Loss_view_freq == 0:
            print(f"Epoch {i + 1}/{epochs}, Loss: {loss:.6f}")
            x_epochs.append(i + 1)
            y_loss.append(float(loss))

    print("訓練が完了しました")
    # print(x_epochs)
    # print(y_loss)
    print(min(y_loss))

# 6. 結果の可視化 (2つのグラフを描画)

# 描画エリア全体を「fig」、各グラフの描画エリアを「axes」として準備
# plt.subplots(行数, 列数, figsize=(全体の横幅, 全体の縦幅))
# 今回は縦に2つ並べるので (2, 1)
fig, axes = plt.subplots(2, 1, figsize=(8, 12))

# --- グラフ1: モデルの予測結果 ---
ax1 = axes[0]  # 1番目の描画エリア
ax1.scatter(x, y_true, s=10, label="True Data ($y=x^2$)")
ax1.plot(x, net.forward(x), color="red", linewidth=3, label="Model Prediction")
ax1.set_title("Neural Network Approximation of $y=x^2$")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.legend()
ax1.grid(True)

# --- グラフ2: 損失の推移 ---
ax2 = axes[1]  # 2番目の描画エリア
ax2.plot(x_epochs, y_loss, color="blue")
ax2.set_title("Training Loss over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss (MSE)")
# y軸を対数スケールにすると、序盤の大きな減少と終盤の細かな変化の両方が見やすくなる
ax2.set_yscale("log")
ax2.grid(True)

# グラフ同士が重ならないようにレイアウトを自動調整
fig.tight_layout()

# グラフを表示
plt.show()

"""勾配再更新を今回はすべて一括で更新しているが、
一個一個更新する手法でも試してみて、
二つの結果を比べることでepochs-lossグラフの違いを調べる。
back_propagationにおいてoutput->hidden_layer(更新パラメーターはweight=3, bias=1の四つ)と
hidden_layer->input(更新パラメーターはweight=3, bias=1の四つ)を学習率をそれぞれ指定して交互に変更させていく学習方法にする
つまり計算は今までの倍かかるが、より確実に丁寧にする方法を採択して学習を最適化してグラフに添わせるようにする。
"""
"""学習率は動的に更新できるようなプログラムにする
（参照するのはepochs-lossでグラフが上昇したらより小さい学習率に代わるみたいな
cf（0.1->0.05->0.01->0.005->…））"""
"""初めて聞く人が読んでわかるように説明できるようにpaperにまとめて（用語説明、機構説明）、
数学的なnotationを定めておく（たとえば記号が全体通して被らないように、
input=xとかoutput=yみたいに数学的な記号を統一して）"""
