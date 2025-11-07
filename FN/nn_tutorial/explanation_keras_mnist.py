# ===============================================
# 手書き数字分類（sklearn digits）× Keras CNN の最小実装（詳細コメント付き）
# ---------------------------------------------------------------
# ・データ    : sklearn の 8x8 グレースケール手書き数字（0〜9）
# ・前処理    : one-hot 化（目的変数）、(N, 64) → (N, 8, 8, 1) へ形状変換（入力）
# ・モデル    : Conv(3x3, ch=5) → Conv(2x2, ch=3) → Flatten → Dense(softmax)
# ・損失/最適化: categorical_crossentropy / SGD
# ・評価      : classification_report（precision, recall, f1, support）
# ＊注意      : `from tensorflow.python import keras as K` は TF の内部APIである。
#               将来互換性の観点では `from tensorflow import keras as K` を推奨だが、
#               本コードでは“元のコードをなるべく消さない”方針で据え置く。
# ===============================================

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import classification_report
from tensorflow.python import keras as K  # ★ 元の指定を維持

# ---------------------------------------------------------------
# 1) データ読み込み
#    - dataset.data : (N, 64) の画素フラット配列（8x8 を一次元化）
#    - dataset.target: (N,) のラベル（0..9）
# ---------------------------------------------------------------
dataset = load_digits()

# CNN 入力のための形状（高さ=8, 幅=8, チャネル=1）
image_shape = (8, 8, 1)
num_class = 10  # クラス数（0〜9）

# ---------------------------------------------------------------
# 2) 目的変数の one-hot 化
#    - 例: 3 → [0,0,0,1,0,0,0,0,0,0]
#    - 一部環境では `K.utils.to_categorical` が内部APIに存在しないため、
#      AttributeError を補足してローカル実装にフォールバックする。
#      （元の1行は残し、動作互換のみを追加）
# ---------------------------------------------------------------
y = dataset.target
try:
    y = K.utils.to_categorical(y, num_class)  # 元の処理（そのまま残す）
except AttributeError:
    # --- 最小限のフォールバック実装（互換 one-hot）---
    y = np.asarray(y, dtype=np.int64).ravel()
    n = y.size
    oh = np.zeros((n, num_class), dtype=np.float32)
    oh[np.arange(n), y] = 1.0
    y = oh

# ---------------------------------------------------------------
# 3) 説明変数の形状変換
#    - (N, 64) → 各サンプルを (8, 8, 1) へ reshape
#    - CNN は (H, W, C) のテンソル入力を前提とする
#    - 必要に応じて画素値を 0-1 スケーリング（例: /16.0）することを推奨
# ---------------------------------------------------------------
X = dataset.data
X = np.array([data.reshape(image_shape) for data in X], dtype=np.float32)
# X = X / 16.0  # ← 正規化したい場合はコメント解除（元の行は変更しない）

# ---------------------------------------------------------------
# 4) 学習/評価データ分割
#    - test_size=0.33: データの 33% を評価用に確保
#    - 再現性が必要なら random_state を固定
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# ---------------------------------------------------------------
# 5) モデル定義（非常に小さな CNN）
#    [Conv2D(5, 3x3, same, ReLU)] → [Conv2D(3, 2x2, same, ReLU)]
#    → [Flatten] → [Dense(num_class, softmax)]
# ---------------------------------------------------------------
model = K.Sequential(
    [
        K.layers.Conv2D(
            filters=5,
            kernel_size=3,
            strides=1,
            padding="same",
            input_shape=image_shape,
            activation="relu",
        ),
        K.layers.Conv2D(
            filters=3,
            kernel_size=2,
            strides=1,
            padding="same",
            activation="relu",
        ),
        K.layers.Flatten(),
        K.layers.Dense(units=num_class, activation="softmax"),
    ]
)

# ---------------------------------------------------------------
# 6) コンパイル設定：one-hot ラベル用の交差エントロピー + SGD
# ---------------------------------------------------------------
model.compile(loss="categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])

# ---------------------------------------------------------------
# 7) 学習：エポック数はチュートリアル用の小回数（必要に応じて調整）
# ---------------------------------------------------------------
model.fit(X_train, y_train, epochs=8, verbose=1)

# ---------------------------------------------------------------
# 8) 推論 → ラベル化（argmax）
# ---------------------------------------------------------------
predicts = model.predict(X_test, verbose=0)
predicts = np.argmax(predicts, axis=1)

# 実ラベル（one-hot → argmax）
actual = np.argmax(y_test, axis=1)

# ---------------------------------------------------------------
# 9) 評価レポート出力
# ---------------------------------------------------------------
print(classification_report(actual, predicts))

# ---------------------------------------------------------------
# 【改善ヒント】
# ・入力正規化（/16.0）で学習安定化。
# ・最適化を Adam に変更、学習率調整、BatchNorm/Dropout の導入。
# ・混同行列の可視化で誤分類傾向を分析。
# ---------------------------------------------------------------
