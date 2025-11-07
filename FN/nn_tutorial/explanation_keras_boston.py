# ===============================================
# Keras + scikit-learn による回帰（Boston Housing）最小実装（詳細コメント付き・修正版）
# ---------------------------------------------------------------
# 変更点（主な修正）:
#  1) 内部APIの使用を廃止: `from tensorflow.python import keras as K` → 公開API `from tensorflow import keras as K`
#     → AttributeError: 'BatchNormalization' が存在しない問題を解消。
#  2) 再現性と安定化: 乱数シード固定、float32 への型統一、冗長ログ抑制を追加（任意）。
#  3) 警告対策: Boston の非推奨警告(FutureWarning)を一時的に無視（※教材用途を前提）。
#  4) 可視化: 軸ラベル、タイトル、45度線（y=x）を追加して評価を視覚的に明確化。
# ---------------------------------------------------------------
# 注意:
#  - Boston データセットは倫理的問題により scikit-learn 1.2 以降で削除予定の非推奨データである。
#    学術的・教材目的でのみ使用し、実験では California/Ames 等の代替データセットを推奨である。
#  - 環境制約によりバージョンを据え置く前提で、動作最小修正のみを行っている。
# ===============================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston  # ※ 非推奨（環境 1.0.x では使用可）

# --- TensorFlow/Keras は公開APIから import（内部API禁止） ---
from tensorflow import keras as K
import tensorflow as tf

# ---------------------------------------------------------------
# （任意）実行時ノイズ抑制・再現性確保
# ---------------------------------------------------------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # TFの冗長ログ抑制（0/1/2/3）
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn"
)  # Bostonの非推奨警告を抑制
np.random.seed(0)
tf.random.set_seed(0)

# ---------------------------------------------------------------
# 1) データ読み込み
#    X: 特徴量(13次元), y: 住宅価格（連続値）
#    ここでは前処理を簡略化（本来は標準化など推奨）
# ---------------------------------------------------------------
dataset = load_boston()
X = dataset.data.astype(np.float32)  # Keras整合のため float32 に統一
y = dataset.target.astype(np.float32)

# ---------------------------------------------------------------
# 2) 学習/評価データ分割
#    test_size=0.33 → 33% をテストに使用
#    random_state で再現性を確保
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0
)

# ---------------------------------------------------------------
# 3) モデル定義（回帰用の全結合ネット）
#    構成: BatchNorm → Dense(softplus + L1) → Dense(1, 線形)
#    ・BatchNormalization: 特徴量のスケール/分布のばらつきを内部で調整し学習安定化を狙う
#    ・softplus: ReLUの平滑近似で勾配が消えにくい。出力は常に正
#    ・kernel_regularizer="l1": L1正則化で疎性（不要重みのゼロ化）を促進
# ---------------------------------------------------------------
model = K.Sequential(
    [
        K.layers.BatchNormalization(input_shape=(13,)),
        K.layers.Dense(units=13, activation="softplus", kernel_regularizer="l1"),
        K.layers.Dense(units=1),  # 回帰なので活性化なし（線形）
    ]
)

# ---------------------------------------------------------------
# 4) コンパイル設定
#    損失: MSE（平均二乗誤差）、最適化: SGD（学習率等は既定値）
#    ※ 収束性を高めたい場合は 'adam' の採用や学習率調整を検討
# ---------------------------------------------------------------
model.compile(loss="mean_squared_error", optimizer="sgd")

# ---------------------------------------------------------------
# 5) 学習
#    エポック数は動作確認用に小さめ（8）。本格評価では early stopping を推奨
# ---------------------------------------------------------------
history = model.fit(X_train, y_train, epochs=8, batch_size=32, verbose=0)

# ---------------------------------------------------------------
# 6) 推論
#    出力 predicts の形状は (テストサンプル数, 1) → 可視化用に (N,) へ変形
# ---------------------------------------------------------------
predicts = model.predict(X_test, verbose=0).reshape(-1)

# ---------------------------------------------------------------
# 7) 可視化（予測 vs 実測）
#    理想的には y = x の直線上に点が分布
# ---------------------------------------------------------------
result = pd.DataFrame({"predict": predicts, "actual": y_test})
limit = float(np.max(y_test))

ax = result.plot.scatter(x="actual", y="predict", xlim=(0, limit), ylim=(0, limit))
ax.set_title("Prediction vs Actual (Boston Housing) — BN + softplus + L1")
ax.set_xlabel("Actual")
ax.set_ylabel("Predict")

# 45度線（y=x）で視覚的な一致度を確認
line = np.linspace(0, limit, 100, dtype=np.float32)
plt.plot(line, line, linestyle="--", linewidth=1.0, color="gray")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# 【理論メモ】
# ・本モデルはアフィン変換と非線形活性化（softplus）を組み合わせた基本的な回帰ネットである。
# ・BatchNorm は内部統計で入力を正規化し、最適化の条件数を改善することで収束を助けることが多い。
# ・L1 正則化は不要な重みを 0 近傍に誘導し、モデルの解釈性や汎化性能の向上に寄与し得る。
# ・本タスクではスケーリング（StandardScaler等）導入で性能が安定しやすい。
#   バージョン据え置きの方針のため本実装では割愛したが、実運用では導入を推奨する。
# ---------------------------------------------------------------
