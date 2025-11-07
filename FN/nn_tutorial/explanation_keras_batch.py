# ===============================================
# Keras（公開API）での2層全結合ネットワーク（GPU対応・最小例）
#  - 非公開API: `from tensorflow.python import keras` は使用しない
#  - 公開API:   `from tensorflow import keras as K` を使用
#  - 目的: バッチ入力 (3, 2) → 出力形状 (3, 4) の確認
# ===============================================

import numpy as np
from tensorflow import keras as K
import tensorflow as tf

# （任意）再現性のためのシード固定
np.random.seed(0)
tf.random.set_seed(0)

# GPU が認識されているかを確認（Metal/MPS 環境なら1つ以上が列挙される）
print("Physical GPUs:", tf.config.list_physical_devices("GPU"))

# ---------------------------------------------------------------
# 2層の全結合ネットワーク
# 入力: 2次元, 隠れ層: 4ユニット + sigmoid, 出力層: 4ユニット（線形）
# ---------------------------------------------------------------
model = K.Sequential(
    [
        K.layers.Dense(units=4, input_shape=(2,), activation="sigmoid"),
        K.layers.Dense(units=4),
    ]
)

# ---------------------------------------------------------------
# バッチサイズ3・特徴量2の入力を作成（float32 推奨）
# ---------------------------------------------------------------
batch = np.random.rand(3, 2).astype(np.float32)

# 推論（学習ではないため verbose=0 で静かに実行）
y = model.predict(batch, verbose=0)

# 期待される出力形状は (3, 4)
print(y.shape)  # -> (3, 4)
