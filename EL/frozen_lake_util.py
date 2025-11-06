# =========================================
# FrozenLake の Q値可視化ユーティリティ（詳細コメント＋理論補足付き）
# -----------------------------------------
# 目的：
#   ・FrozenLake の各セルを 3x3 の小グリッドで描画し、
#     上下左右（UP/DOWN/LEFT/RIGHT）の Q値と、その平均（中央）を色で可視化する。
#
# 理論背景（要点）：
#   ・Q学習では、行動価値関数 Q(s,a) を推定し、方策は通常 argmax_a Q(s,a)（貪欲）や
#     ε-greedy 等で選択する。
#   ・FrozenLake の行動は {LEFT=0, DOWN=1, RIGHT=2, UP=3}（Gym 標準）。
#   ・この可視化では、状態 s の 4 方向の Q(s,a) を
#       上：UP(3), 下：DOWN(1), 左：LEFT(0), 右：RIGHT(2), 中央：平均
#     にマッピングして色（RdYlGn）で表示する。
#
# 使い方（例）：
#   Q = np.zeros((16, 4))                # 4x4 FrozenLake の Q テーブル
#   show_q_value(Q, env_id="FrozenLake-v1")
#
# 備考：
#   ・描画レンジ（vmin/vmax）は与えられた Q の絶対最大値に合わせて対称にスケーリング。
#   ・Q が dict の場合は {state: [Q_left, Q_down, Q_right, Q_up], ...} を想定。
# =========================================

from typing import Union, Dict, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Gym の登録（is_slippery=False の “Easy” 環境を追加で用意）
try:
    import gym
    from gym.envs.registration import register

    register(
        id="FrozenLakeEasy-v0",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={"is_slippery": False},
    )
except Exception:
    # 既に登録済み等の例外は握りつぶして続行（環境が使えればOK）
    pass


def _ensure_numpy_row(Qs: Union[Sequence[float], np.ndarray]) -> np.ndarray:
    """Q(s,·) を 1次元の numpy 配列に変換（長さ4を想定）。長さが足りない場合は例外。"""
    arr = np.asarray(Qs, dtype=float).reshape(-1)
    if arr.size < 4:
        raise ValueError(f"Q(s,·) must have 4 actions, got shape={arr.shape}.")
    return arr


def show_q_value(
    Q: Union[np.ndarray, Dict[int, Sequence[float]]],
    env_id: str = "FrozenLake-v1",
    cmap=cm.RdYlGn,
    interpolation: str = "bilinear",
) -> None:
    """
    FrozenLake 系環境の Q値を 3x3 マトリクスで可視化する。

    引数:
        Q      : 形状 (num_states, 4) の ndarray あるいは {state: [L, D, R, U]} の辞書
        env_id : "FrozenLake-v1" / "FrozenLakeEasy-v0" など
        cmap   : matplotlib のカラーマップ
        interpolation : imshow の補間方法

    出力:
        Matplotlib 図（plt.show() で表示）
    """
    # --- 環境のメタ情報を取得（行列サイズなど） ---
    try:
        env = gym.make(env_id)
    except gym.error.DeprecatedEnv:
        # Gym >=0.26 では FrozenLake-v0 が削除されたため、-v1 にフォールバック
        if env_id.endswith("-v0"):
            fallback_id = env_id[:-3] + "-v1"
            env = gym.make(fallback_id)
            env_id = fallback_id
        else:
            raise
    nrow = int(env.unwrapped.nrow)  # 例：4
    ncol = int(env.unwrapped.ncol)  # 例：4
    state_size = 3  # 1セルを 3x3 で描画
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size

    # --- 可視化用キャンバス（Q値の配置面） ---
    reward_map = np.zeros((q_nrow, q_ncol), dtype=float)

    # Q が ndarray の場合のユーティリティ（範囲ガード）
    def _state_exists(s: int) -> bool:
        if isinstance(Q, dict):
            return s in Q
        elif isinstance(Q, (np.ndarray, np.generic)):
            return 0 <= s < Q.shape[0]
        else:
            return False

    # --- 各セルの 3x3 へ Q を配置 ---
    # 表示上は y 軸が上に行くほどプラスになるため、行インデックスを反転して描画
    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            if not _state_exists(s):
                continue

            # 3x3 の中心座標（描画用）: 行は反転して配置
            _r = 1 + (nrow - 1 - r) * state_size
            _c = 1 + c * state_size

            q_s = _ensure_numpy_row(Q[s] if isinstance(Q, dict) else Q[s])

            # Gym の FrozenLake 標準アクション定義に合わせる：
            # LEFT=0, DOWN=1, RIGHT=2, UP=3
            left, down, right, up = q_s[0], q_s[1], q_s[2], q_s[3]
            center = float(np.mean(q_s[:4]))

            # 3x3 マトリクスへ配置
            reward_map[_r][_c - 1] = left  # 左
            reward_map[_r - 1][_c] = down  # 下（座標反転に注意：描画上は上方向が正）
            reward_map[_r][_c + 1] = right  # 右
            reward_map[_r + 1][
                _c
            ] = up  # 上（描画上で下側に置くと視覚的に上矢印に対応）
            reward_map[_r][_c] = center  # 中央（平均値）

    # --- カラースケール：±max(|Q|) に対称化（全ゼロのときは 1.0 を使用） ---
    absmax = float(np.nanmax(np.abs(reward_map))) if reward_map.size else 0.0
    if absmax == 0.0 or not np.isfinite(absmax):
        absmax = 1.0

    # --- 描画 ---
    fig = plt.figure(figsize=(max(5, ncol), max(5, nrow)))
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(
        reward_map,
        cmap=cmap,
        interpolation=interpolation,
        vmax=absmax,
        vmin=-absmax,
        origin="upper",  # 上が0行になる向き（デフォルト）
    )

    # セル境界線（太線）を 3 つおきに表示
    ax.set_xlim(-0.5, q_ncol - 0.5)
    ax.set_ylim(-0.5, q_nrow - 0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))

    # 軸ラベルはグリッド座標（0..ncol, 0..nrow）を表示
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))

    # 細線グリッド（全マス目）も重ねる
    ax.set_xticks(np.arange(-0.5, q_ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, q_nrow, 1), minor=True)
    ax.grid(which="major", color="k", linewidth=1.0, alpha=0.6)
    ax.grid(which="minor", color="k", linewidth=0.2, alpha=0.2)

    # カラーバーとタイトル
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Q-value")
    ax.set_title(
        f"FrozenLake Q-values ({env_id})\n"
        "Center: mean of [L, D, R, U] / Edges: each action value"
    )

    plt.tight_layout()
    plt.show()


# 参考：最小実行例（コメントアウト）
# Q = np.random.randn(16, 4) * 0.1  # 例：ランダムな小さめの Q 値
# show_q_value(Q, env_id="FrozenLakeEasy-v0")
