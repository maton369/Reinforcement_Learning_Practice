# =========================================================
# ε-greedy 学習ログ可視化用エージェント（解説コメント＋理論補足付き）
# ---------------------------------------------------------
# ・役割：
#     - 行動選択：ε-greedy（確率 ε で探索、1-ε で推定最良行動）
#     - 価値表 Q[s][a] を保持（辞書：状態→行動価値配列）
#     - 報酬ログを集計・区間平均±標準偏差を可視化
#
# 【理論背景（要点）】
#   - ε-greedy：探索（exploration）と活用（exploitation）のバランスをとる単純な方策。
#       π(a|s) = ε / |A(s)|   （探索時は一様）
#                else argmax_a Q(s,a)
#   - 可視化：一定区間（interval）ごとに報酬の平均と標準偏差を描画し、学習の安定度を把握。
#   - 注意：このクラスは「方策」および「ログ可視化」を提供する最小構成であり、
#           Q 値の更新（例：Q学習、SARSA）は別途学習ループ側で行う想定。
# =========================================================

from typing import Dict, List, Sequence, Optional
import numpy as np
import matplotlib.pyplot as plt


class ELAgent:
    """
    Exploration-Logging Agent（方策＋ログ可視化用の軽量ユーティリティ）
    - Q:  状態 s をキー、値として各行動の推定価値配列（list/ndarray）を格納する辞書
    - ε:  探索率（0〜1）
    - reward_log: 時系列のスカラー報酬を保持（可視化対象）
    """

    def __init__(self, epsilon: float) -> None:
        self.Q: Dict[object, List[float]] = {}
        self.epsilon: float = float(epsilon)
        self.reward_log: List[float] = []

    # --------------------------------
    # 乱数の再現性を確保したい場合に使用
    # --------------------------------
    def seed(self, seed: int) -> None:
        np.random.seed(seed)

    # --------------------------------
    # ε-greedy 方策
    #   - s が未知 or Q[s] が未初期化（総和=0等）ならランダム
    #   - それ以外は argmax(Q[s])
    # --------------------------------
    def policy(self, s: object, actions: Sequence[object]) -> int:
        """
        引数:
            s       : 現在の状態（ハッシュ可能な任意オブジェクトを想定）
            actions : 取りうる行動のリスト/タプル（len(actions) で次元を取得）
        戻り値:
            選択した行動のインデックス（0..len(actions)-1）
        """
        n_actions = len(actions)
        if n_actions == 0:
            raise ValueError("actions is empty.")

        # 探索
        if np.random.random() < self.epsilon:
            return int(np.random.randint(n_actions))

        # 活用：既知状態で Q が有効なら argmax、そうでなければランダム
        if s in self.Q and len(self.Q[s]) == n_actions and np.sum(self.Q[s]) != 0:
            return int(np.argmax(self.Q[s]))
        else:
            return int(np.random.randint(n_actions))

    # --------------------------------
    # ログ操作
    # --------------------------------
    def init_log(self) -> None:
        """報酬ログをリセットする。"""
        self.reward_log = []

    def log(self, reward: float) -> None:
        """報酬をタイムステップ順に追記する。"""
        self.reward_log.append(float(reward))

    # --------------------------------
    # 報酬ログの可視化
    #   - episode > 0 の場合：末尾 interval 分の統計を print
    #   - それ以外：0, interval, 2*interval, ... ごとに区間平均・標準偏差を帯で描画
    # --------------------------------
    def show_reward_log(self, interval: int = 50, episode: int = -1) -> None:
        """
        引数:
            interval : 区間幅（この幅ごとに区間平均・標準偏差をとる）
            episode  : >0 のとき、末尾 interval の統計を print（図は描かない）
        """
        if interval <= 0:
            raise ValueError("interval must be positive.")

        T = len(self.reward_log)
        if T == 0:
            print("No rewards to show.")
            return

        # 末尾の統計だけを表示
        if episode > 0:
            # 末尾が interval 未満しかなければ、可能な範囲を使う
            L = min(interval, T)
            rewards = self.reward_log[-L:]
            mean = float(np.round(np.mean(rewards), 3))
            std = float(np.round(np.std(rewards), 3))
            print(
                f"At Episode {episode} average reward over last {L} steps is {mean} (+/-{std})."
            )
            return

        # 0..T を interval でスライスして区間統計を取る
        indices = list(range(0, T, interval))
        means, stds = [], []
        for i in indices:
            chunk = self.reward_log[i : (i + interval)]
            means.append(float(np.mean(chunk)))
            stds.append(float(np.std(chunk)))

        means = np.asarray(means)
        stds = np.asarray(stds)

        # 可視化
        plt.figure()
        plt.title("Reward History")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.fill_between(
            indices,
            means - stds,
            means + stds,
            alpha=0.15,
            color="g",
            label="mean ± std",
        )
        plt.plot(indices, means, "o-", color="g", label=f"mean per {interval} steps")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()


# =========================
# 使い方（参考・疑似コード）
# -------------------------
# env.reset()
# agent = ELAgent(epsilon=0.1)
# agent.init_log()
# s = env.reset()
# done = False
# while not done:
#     a = agent.policy(s, actions=env.actions(s))
#     s_next, r, done, info = env.step(a)
#     agent.log(r)
#     # Q学習などの更新は別途：
#     #   Q[s][a] ← Q[s][a] + α ( r + γ max_a' Q[s_next][a'] - Q[s][a] )
#     s = s_next
# agent.show_reward_log(interval=50)
# =========================
