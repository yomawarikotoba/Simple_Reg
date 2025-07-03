import numpy as np


class MSELoss:
    def forward(self, y_pred, y_true):
        # 損失を計算（誤差関数は平均二乗和誤差）
        # 平均二乗誤差を利用する理由は誤差や勾配のスケールを安定させ、学習の安定化や学習率の調整をしやすくするため。
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred, y_true):
        num_samples = y_true.shape[0]
        # 逆伝播の最初のステップとして、予測値に関する損失の勾配を返す
        # 全体の誤差関数をt_predで微分した結果が以下
        return 2 * (y_pred - y_true) / num_samples
