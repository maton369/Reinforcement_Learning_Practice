# 強化学習：Q-learning vs SARSA の比較エージェント（詳細コメント付き）
# ---------------------------------------------------------------------
# 目的：
#   - 同一コードパスで Q-learning（off-policy）と SARSA（on-policy）を切替比較する。
#   - FrozenLakeEasy-v0（確定遷移）で学習し、Q値ヒートマップで可視化する。
#
# 重要ポイント：
#   - Q-learning の TDターゲット： r + γ * max_{a'} Q(s', a')
#     → 学習は「最良行動を仮定」するため攻め（リスク許容）になりやすい。
#   - SARSA の TDターゲット： r + γ * Q(s', a')（a' は実際に方策からサンプル）
#     → 学習は「実際の方策」を反映するため守り（リスク回避）になりやすい。
#
# 実装メモ：
#   - ε-greedy 方策は ELAgent に実装済み（policy(s, actions)）。
#   - 失敗終端（done 且つ reward==0）に対しペナルティ報酬を与えて回避嗜好を強めることができる。
#   - multiprocessing で両手法を並列に実行し、可視化を続けて行う。
#
# 既知のAPI差：
#   - ここでは旧Gym API（reset()->obs, step()->(obs, reward, done, info)）を前提とする。
#   - 新APIでは reset() が (obs, info)、step() が (obs, reward, terminated, truncated, info) になる点に注意。

from multiprocessing import Pool
from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class CompareAgent(ELAgent):
    """
    Q-learning と SARSA を切替可能な比較エージェントである。

    Attributes
    ----------
    q_learning : bool
        True のとき Q-learning（off-policy）、False のとき SARSA（on-policy）。
    epsilon : float
        ε-greedy の探索率（ELAgent 側で利用）。
    """

    def __init__(
        self, q_learning: bool = True, epsilon: float = 0.33, fail_penalty: float = -0.5
    ):
        self.q_learning = q_learning  # 手法切替フラグ
        self.fail_penalty = fail_penalty  # 失敗終端ペナルティ
        super().__init__(epsilon)  # 方策とログ機能を継承

    def learn(
        self,
        env: gym.Env,
        episode_count: int = 1000,
        gamma: float = 0.9,
        learning_rate: float = 0.1,
        render: bool = False,
        report_interval: int = 50,
    ) -> None:
        """
        Q-learning/SARSA のいずれかで学習を実施する。

        TDターゲットの違い：
            Q-learning:  r + γ * max_{a'} Q(s', a')
            SARSA:       r + γ * Q(s', a')   （a' は方策からのサンプル）
        """
        # 学習曲線ログを初期化
        self.init_log()

        # 行動集合を先に確定させる（これに依存して Q の初期化を行う）
        actions = list(range(env.action_space.n))

        # Q テーブル（疎構造）：未知状態に初アクセス時、長さ|A|の0配列を生成
        self.Q = defaultdict(lambda: [0.0] * len(actions))

        # ===== エピソード反復 =====
        for e in range(episode_count):
            s = env.reset()
            done = False

            # 初手の行動をサンプル（SARSA は on-policy のため、初期 a が必要）
            a = self.policy(s, actions)

            # --- タイムステップ反復 ---
            while not done:
                if render:
                    env.render()

                # 行動 a を実行 → 次状態・報酬・終端を取得
                n_state, reward, done, info = env.step(a)

                # 失敗終端（報酬0）に軽いペナルティを与えて危険経路を抑制（設計選択）
                if done and reward == 0:
                    reward = self.fail_penalty

                # 次状態での行動（on-policy 用）。Q-learning の場合も計算上は用意しておく。
                n_action = self.policy(n_state, actions)

                # ---------------------------
                # TDターゲットの選択
                # ---------------------------
                if self.q_learning:
                    # off-policy: 次状態で最良の行動価値を仮定（max）
                    gain = reward + gamma * max(self.Q[n_state])
                else:
                    # on-policy: 実際にサンプルした次行動 a' の価値を使用
                    gain = reward + gamma * self.Q[n_state][n_action]

                # 現在の推定値
                estimated = self.Q[s][a]

                # 1ステップ更新：Q ← Q + α (gain - estimated)
                self.Q[s][a] += learning_rate * (gain - estimated)

                # 次状態へ遷移
                s = n_state

                # 次の行動選択
                if self.q_learning:
                    # Q-learning は次行動を「その都度」選び直す（方策は行動選択にのみ使用）
                    a = self.policy(s, actions)
                else:
                    # SARSA は on-policy の鎖を維持： (s,a) ← (s',a')
                    a = n_action

            else:
                # エピソード末尾の報酬（成功=1, 失敗=ペナルティ）をログに記録
                self.log(reward)

            # 規定間隔で学習進捗を表示（直近区間の平均±標準偏差）
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train(q_learning: bool):
    """
    単一プロセスで手法を指定して学習を実施し、学習後の Q テーブルを辞書にして返す。
    q_learning=True  → Q-learning
    q_learning=False → SARSA
    """
    env = gym.make("FrozenLakeEasy-v0")  # 確定遷移で安定学習
    agent = CompareAgent(q_learning=q_learning)  # ε は既定 0.33
    agent.learn(env, episode_count=3000)  # ある程度長めに回して差異を強調
    return dict(agent.Q)  # multiprocessing で送れる形に変換


if __name__ == "__main__":
    # 2手法（Q-learning, SARSA）を並列に学習
    with Pool() as pool:
        # True=Q-learning, False=SARSA
        results = pool.map(train, [True, False])

    # 結果を順に可視化（環境IDは学習と同じものを明示）
    labels = ["Q-learning (off-policy)", "SARSA (on-policy)"]
    for label, Q in zip(labels, results):
        print(f"[Visualization] {label}")
        show_q_value(Q, env_id="FrozenLakeEasy-v0")
