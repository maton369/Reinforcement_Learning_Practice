# ===========================================
# EL/monte_carlo.py （修正後）
#  - 可視化呼び出しで env_id="FrozenLakeEasy-v0" を明示
# ===========================================
import math
from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class MonteCarloAgent(ELAgent):
    """
    モンテカルロ法（on-policy, every-visit）の実装クラス
    ---------------------------------------------------------
    ・ε-greedy 方策を用いて環境と相互作用し、エピソード単位で学習を行う。
    ・各エピソードで得た報酬系列から割引累積報酬 G を算出し、
      その平均を Q(s,a) に逐次的に反映する。
    """

    def __init__(self, epsilon=0.1):
        # 親クラス（ELAgent）からε-greedy 方策とログ機能を継承
        super().__init__(epsilon)

    def learn(
        self, env, episode_count=1000, gamma=0.9, render=False, report_interval=50
    ):
        """
        モンテカルロ法による学習を実行する。

        Parameters
        ----------
        env : gym.Env
            学習対象の環境（例：FrozenLake）
        episode_count : int
            エピソード数
        gamma : float
            割引率
        render : bool
            True の場合、環境を可視化
        report_interval : int
            指定間隔で学習進捗（平均報酬）を出力
        """
        self.init_log()  # 報酬ログを初期化
        actions = list(range(env.action_space.n))  # 環境の行動空間を取得
        self.Q = defaultdict(lambda: [0] * len(actions))  # 各 (s,a) の Q 値を初期化
        N = defaultdict(lambda: [0] * len(actions))  # 各 (s,a) の訪問回数を記録

        # ===============================
        # 各エピソードの学習ループ
        # ===============================
        for e in range(episode_count):
            s = env.reset()  # 状態を初期化（旧Gym API想定）
            done = False
            experience = []  # エピソード内の履歴を保存

            # -------------------------------
            # 1エピソードを完了するまで行動
            # -------------------------------
            while not done:
                if render:
                    env.render()  # 描画（デバッグ用）

                # ε-greedy 方策に基づき行動を選択
                a = self.policy(s, actions)

                # 環境に行動を適用 → 次状態・報酬・終了フラグを取得
                n_state, reward, done, info = env.step(a)

                # 状態・行動・報酬を保存
                experience.append({"state": s, "action": a, "reward": reward})

                # 次の状態へ遷移
                s = n_state
            else:
                # whileを正常終了（breakなし）した場合に報酬をログ
                self.log(reward)

            # -------------------------------
            # エピソード終了後：学習（MC評価）
            # -------------------------------
            for i, x in enumerate(experience):
                s, a = x["state"], x["action"]

                # (s,a)以降の割引累積報酬 G を計算
                G, t = 0, 0
                for j in range(i, len(experience)):
                    G += math.pow(gamma, t) * experience[j]["reward"]
                    t += 1

                # (s,a) の訪問回数をカウント
                N[s][a] += 1

                # 学習率 α = 1 / N(s,a)
                alpha = 1 / N[s][a]

                # モンテカルロ法の更新式：
                # Q(s,a) ← Q(s,a) + α [ G - Q(s,a) ]
                self.Q[s][a] += alpha * (G - self.Q[s][a])

            # -------------------------------
            # 進捗報告（一定間隔ごと）
            # -------------------------------
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train():
    """
    FrozenLakeEasy-v0 環境を用いて MonteCarloAgent を学習させ、
    Q値分布と報酬履歴を可視化する。
    """
    agent = MonteCarloAgent(epsilon=0.1)
    env = gym.make("FrozenLakeEasy-v0")

    # 学習実行（エピソード数 500）
    agent.learn(env, episode_count=500)

    # 学習後の Q 値を可視化（★ 学習と同じ環境IDを明示）
    show_q_value(agent.Q, env_id="FrozenLakeEasy-v0")

    # 報酬履歴をプロット
    agent.show_reward_log()


# スクリプトとして実行された場合に学習を開始
if __name__ == "__main__":
    train()


# ===========================================
# EL/frozen_lake_util.py （修正後）
#  - show_q_value のデフォルト env_id を 'FrozenLake-v1' に変更
#  - env 作成に失敗した場合のフォールバック実装を追加
#  - 全ゼロ配列時の vmin/vmax をガード
# ===========================================
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gym

# is_slippery=False の確定遷移版を登録（既登録なら例外を握りつぶす）
try:
    from gym.envs.registration import register

    register(
        id="FrozenLakeEasy-v0",
        entry_point="gym.envs.toy_text:FrozenLakeEnv",
        kwargs={"is_slippery": False},
    )
except Exception:
    pass


def show_q_value(Q, env_id: str = "FrozenLake-v1"):
    """
    FrozenLake の Q値テーブルを 3x3 マスで可視化する。
    中央は各行動Qの平均、上下左右は UP/DOWN/LEFT/RIGHT の値を配置する。

    Parameters
    ----------
    Q : dict[int, seq[float]] | np.ndarray
        形状 (num_states, 4) の配列、または {state: [L, D, R, U]} の辞書
    env_id : str
        使用する FrozenLake 環境ID（既定は v1）。例: "FrozenLakeEasy-v0"
    """
    # gym のバージョン差を吸収：指定 env_id が無ければ順にフォールバック
    env = None
    try:
        env = gym.make(env_id)
    except Exception:
        for fallback in ("FrozenLake-v1", "FrozenLakeEasy-v0", "FrozenLake-v0"):
            try:
                env = gym.make(fallback)
                break
            except Exception:
                continue
    if env is None:
        raise RuntimeError(
            f"Cannot create FrozenLake env. Tried: {env_id}, FrozenLake-v1, FrozenLakeEasy-v0, FrozenLake-v0"
        )

    nrow = env.unwrapped.nrow
    ncol = env.unwrapped.ncol
    state_size = 3
    q_nrow = nrow * state_size
    q_ncol = ncol * state_size
    reward_map = np.zeros((q_nrow, q_ncol), dtype=float)

    # Q が存在するかの判定
    def state_exists(s: int) -> bool:
        if isinstance(Q, dict):
            return s in Q
        elif isinstance(Q, (np.ndarray, np.generic)):
            return 0 <= s < Q.shape[0]
        else:
            return False

    for r in range(nrow):
        for c in range(ncol):
            s = r * ncol + c
            if not state_exists(s):
                continue

            # 表示上の行は上を小さくするため反転
            _r = 1 + (nrow - 1 - r) * state_size
            _c = 1 + c * state_size

            q_s = np.asarray(
                Q[s] if isinstance(Q, dict) else Q[s], dtype=float
            ).reshape(-1)
            if q_s.size < 4:
                continue

            # Gym 標準：LEFT=0, DOWN=1, RIGHT=2, UP=3
            left, down, right, up = q_s[0], q_s[1], q_s[2], q_s[3]
            center = float(np.mean(q_s[:4]))

            reward_map[_r][_c - 1] = left
            reward_map[_r - 1][_c] = down
            reward_map[_r][_c + 1] = right
            reward_map[_r + 1][_c] = up
            reward_map[_r][_c] = center

    absmax = float(np.abs(reward_map).max())
    if absmax == 0.0 or not np.isfinite(absmax):
        absmax = 1.0  # 全ゼロでも表示できるようにスケール確保

    fig = plt.figure(figsize=(max(5, ncol), max(5, nrow)))
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(
        reward_map,
        cmap=cm.RdYlGn,
        interpolation="bilinear",
        vmax=absmax,
        vmin=-absmax,
        origin="upper",
    )

    # 3マスごとの太線グリッド
    ax.set_xlim(-0.5, q_ncol - 0.5)
    ax.set_ylim(-0.5, q_nrow - 0.5)
    ax.set_xticks(np.arange(-0.5, q_ncol, state_size))
    ax.set_yticks(np.arange(-0.5, q_nrow, state_size))
    ax.set_xticklabels(range(ncol + 1))
    ax.set_yticklabels(range(nrow + 1))

    # 細線の補助グリッド
    ax.set_xticks(np.arange(-0.5, q_ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, q_nrow, 1), minor=True)
    ax.grid(which="major", color="k", linewidth=1.0, alpha=0.6)
    ax.grid(which="minor", color="k", linewidth=0.2, alpha=0.2)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Q-value")
    ax.set_title(
        f"FrozenLake Q-values ({env.spec.id})\n"
        "Center: mean of [L, D, R, U] / Edges: each action value"
    )

    plt.tight_layout()
    plt.show()
