# 強化学習：SARSA（on-policy TD学習）の最小実装（詳細コメント付き）
# --------------------------------------------------------------------
# ・方策：ε-greedy（探索：確率 ε、活用：1-ε）— ELAgent に実装済みの policy() を使用
# ・更新式：Q(s,a) ← Q(s,a) + α { r + γ Q(s',a') − Q(s,a) }
#   - TDターゲットは「次状態 s' で、実際に方策からサンプルした行動 a' のQ値」
#   - よって on-policy（方策評価と改善が同じ方策上で進む）
# ・FrozenLakeEasy-v0（確定遷移）での教育用デモを想定
# ・注意：Gym の旧/新API差（reset/step の戻り値）があるため、環境により調整すること
#   - 本コードは旧API（reset()->obs, step()->(obs, reward, done, info)）を例示
# --------------------------------------------------------------------

from collections import defaultdict
import gym
from el_agent import ELAgent
from frozen_lake_util import show_q_value


class SARSAAgent(ELAgent):
    """
    SARSA エージェント（on-policy TD(0)）
    - Qテーブルは defaultdict で疎に保持： state -> [Q_L, Q_D, Q_R, Q_U]
    - 行動選択は ELAgent の ε-greedy に委譲（policy(s, actions)）
    - 学習は 1 ステップ毎に TD 誤差で更新
    """

    def __init__(self, epsilon: float = 0.1):
        # 探索率 ε を基底クラスに渡す（ログ機能・可視化ヘルパも継承される）
        super().__init__(epsilon)

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
        SARSA による on-policy 学習ループである。

        Args:
            env: Gym 互換の離散環境（例：FrozenLake）
            episode_count: 総エピソード数
            gamma: 割引率 γ（0 ≤ γ < 1）
            learning_rate: 学習率 α
            render: True のとき環境を描画（旧 render API を想定）
            report_interval: 学習経過を標準出力する間隔（エピソード数）
        """
        # 学習曲線用ログを初期化
        self.init_log()

        # 行動集合を [0, 1, ..., n-1] として用意
        actions = list(range(env.action_space.n))

        # Qテーブルを遅延初期化（未知の状態アクセス時に 0 で長さ |A| の配列を生成）
        self.Q = defaultdict(lambda: [0.0] * len(actions))

        # ===== エピソード反復 =====
        for e in range(episode_count):
            # 初期状態（Gym 新APIでは (obs, info) = env.reset() になる点に注意）
            s = env.reset()
            done = False

            # SARSA は on-policy なので、状態 s に対する「最初の行動 a」を先にサンプルしておく
            a = self.policy(s, actions)

            # --- タイムステップ反復：終端まで ---
            while not done:
                if render:
                    env.render()

                # 現在の (s, a) を実行して、次状態・報酬・終端を得る
                n_state, reward, done, info = env.step(a)

                # on-policy：次状態 n_state でも「現在の方策」に従って行動を選ぶ（a'）
                n_action = self.policy(n_state, actions)

                # ---------------------------
                # SARSA の TD 更新
                # ---------------------------
                # TDターゲット（gain）は r + γ * Q(s', a')（次状態で実際に選んだ a' を使う）
                gain = reward + gamma * self.Q[n_state][n_action]

                # 現在の推定値 Q(s,a)
                estimated = self.Q[s][a]

                # 1ステップ更新：Q ← Q + α (gain - estimated)
                self.Q[s][a] += learning_rate * (gain - estimated)

                # 状態と行動を次に進める（on-policy の鎖： (s,a) ← (s',a') ）
                s = n_state
                a = n_action

            else:
                # while を break せず正常終了したときに実行（Python の while-else 構文）
                # FrozenLake では報酬が疎（成功=1）なので、最終報酬をログする
                self.log(reward)

            # 規定間隔で学習曲線の統計を表示（直近 interval の平均±標準偏差）
            if e != 0 and e % report_interval == 0:
                self.show_reward_log(episode=e)


def train() -> None:
    """
    デモ実行関数：
    - FrozenLakeEasy-v0（確定遷移）で 500 エピソード学習
    - 学習後に Q 値分布（空間マップ）と学習曲線を表示
    """
    agent = SARSAAgent(epsilon=0.1)  # 探索率 ε。将来的に減衰スケジュール(GLIE)も検討可
    env = gym.make("FrozenLakeEasy-v0")  # is_slippery=False で安定学習

    # 学習を実行（γ=0.9, α=0.1 は FrozenLake では典型的かつ安定）
    agent.learn(env, episode_count=500, gamma=0.9, learning_rate=0.1)

    # Q値ヒートマップを表示（学習に用いた環境IDを明示）
    show_q_value(agent.Q, env_id="FrozenLakeEasy-v0")

    # 報酬履歴（区間平均±標準偏差）を表示
    agent.show_reward_log()


if __name__ == "__main__":
    train()
