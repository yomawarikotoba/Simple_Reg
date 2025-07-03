class Network:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        # ネットワークに層を追加
        self.layers.append(layer)

    def forward(self, inputs):
        # ネットワーク全体の順伝播
        # ここではlayersに追加されたlayerを順にそのforward()を回して全体の順伝播を構成している
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, upstream_gradient):
        # ネットワーク全体の逆伝播
        grad = upstream_gradient
        # reversed()は組み込み関数の一つでリストやarrayを受け取り、その要素を逆順で一つずつ取り出すメソッド
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, learning_rate):
        # 勾配降下法によるパラメータの更新
        for layer in self.layers:
            layer.weights -= learning_rate * layer.weights_gradient
            layer.biases -= learning_rate * layer.biases_gradient
