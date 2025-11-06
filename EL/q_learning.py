# 強化学習：Q-learning（オフポリシーTD学習）の最小実装
# ------------------------------------------------------------
# ・方策：ε-greedy（探索：確率 ε、活用：1-ε）
# ・更新式：Q(s,a) ← Q(s,a) + α { r + γ max_{a'} Q(s',a') − Q(s,a) }
#   - TD誤差 δ = [r + γ max_{a'} Q(s',a')] − Q(s,a)
#   - 学習率 α = learning_rate
# ・on-policy ではなく off-policy（ターゲットは常に greedy な max_{a'}）
# ・FrozenLakeEasy-v0（確定遷移）での教育用デモ向け
#   ※ Gym のバージョンによって reset/step の戻り値が異なる点に注意。
#     ここでは旧API（reset()->obs, step()->(obs, reward, done, info)）を想定する。

from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class QLearningAgent(ELAgent):
    """
    Q-learning エージェントである。
    - Qテーブル：defaultdict(state -> [Q(s,LEFT), Q(s,DOWN), Q(s,RIGHT), Q(s,UP)])
    - 行動選択：ELAgent の ε-greedy policy() を使用
    - 学習：1ステップごとに TD 誤差で Q を更新（オフポリシー）
    """

    def __init__(self, epsilon=0.1):
        # 探索率 ε を基底クラスに設定（ログ管理・可視化機能も継承する）
        super().__init__(epsilon)

    def learn(
        self,
        env,
        episode_count: int = 1000,
        gamma: float = 0.9,
        learning_rate: float = 0.1,
        render: bool = False,
        report_interval: int = 50,
    ) -> None:
        """
        Q-learning による学習を実行する。

        Args:
            env: Gym 互換の離散環境（FrozenLake など）
            episode_count: 学習エピソード数
            gamma: 割引率 γ（0≤γ<1）
            learning_rate: 学習率 α
            render: True のとき各ステップ描画（旧 render API を想定）
            report_interval: 進捗表示の間隔（エピソード数）
        """
        # 学習曲線用ログを初期化
        self.init_log()

        # 離散行動空間のサイズを取得し、Qを疎に初期化
        actions = list(range(env.action_space.n))
        self.Q = defaultdict(lambda: [0.0] * len(actions))

        # ===== エピソード反復 =====
        for e in range(episode_count):
            s = env.reset()  # 初期状態（新Gymでは obs, info = env.reset()）
            done = False

            # --- タイムステップ反復（エピソード終端まで）---
            while not done:
                if render:
                    env.render()

                # ε-greedy で行動 a を選択（探索/活用の切り替えは ELAgent.policy による）
                a = self.policy(s, actions)

                # 環境を1ステップ進める（旧API：obs, reward, done, info）
                n_state, reward, done, info = env.step(a)

                # ---------------------------
                # Q-learning の TD 更新
                # ---------------------------
                # 目標値（ターゲット）：r + γ * max_{a'} Q(s', a')
                gain = reward + gamma * max(self.Q[n_state])
                # 現在推定値：Q(s, a)
                estimated = self.Q[s][a]
                # TD誤差に基づく更新：Q ← Q + α (gain - estimated)
                self.Q[s][a] += learning_rate * (gain - estimated)

                # 次状態へ遷移
                s = n_state

            else:
                # while を break せず正常終了したときに実行（Python の while-else 構文）
                # FrozenLake のような疎な報酬タスクでは、最終報酬（成功=1, 失敗=0）をログに記録
                self.log(reward)

            # 規定間隔で学習曲線の統計を標準出力へ
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train() -> None:
    """
    デモ実行：
    - FrozenLakeEasy-v0（確定遷移版）で 500 エピソード学習
    - 学習後に Q マップと学習曲線を表示
    """
    agent = QLearningAgent(epsilon=0.1)  # 探索率 ε（固定）。GLIE化は将来拡張
    env = gym.make("FrozenLakeEasy-v0")  # is_slippery=False で安定学習

    # 学習実行
    agent.learn(env, episode_count=500, gamma=0.9, learning_rate=0.1)

    # Q値分布を可視化（学習に用いた環境IDを明示）
    show_q_value(agent.Q, env_id="FrozenLakeEasy-v0")

    # 報酬履歴を可視化（区間平均±標準偏差帯）
    agent.show_reward_log()


if __name__ == "__main__":
    train()
