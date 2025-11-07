import argparse
import numpy as np
from collections import defaultdict, Counter
import gym
from gym.envs.registration import register

# FrozenLake の転び属性（is_slippery）を無効化した簡易版環境を登録する。
# 迷路は同じだが確率遷移がなくなるため、価値更新の学習が安定しやすい。
register(
    id="FrozenLakeEasy-v0",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"is_slippery": False},
)


class DynaAgent:
    """
    Dyna-Q 風の手法に基づくエージェントである。
    実環境で得た経験により Q 値（ここでは self.value[state][action]）を更新しつつ、
    併せて環境モデル（遷移と報酬の近似）を用いて「想像上の経験（model-based simulation）」でも
    追加更新を行うことで、サンプル効率を高める狙いがある。
    """

    def __init__(self, epsilon=0.1):
        # ε-greedy のεである。探索（ランダム行動）割合を規定する。
        self.epsilon = epsilon
        # 行動集合は学習開始時に環境から決定する（整数ID列）。
        self.actions = []
        # 価値表（Q値）: dict[state] -> list[Q(a)] を保持する。
        self.value = None

    def policy(self, state):
        """
        ε-greedy 方策を実装する。
        - 確率 ε で一様ランダムな行動を選ぶ（探索）。
        - それ以外は、Q値が未更新（和が0）の場合はランダム、
          更新済みであれば argmax により貪欲に選択（活用）する。
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            if sum(self.value[state]) == 0:
                # まだ一度も価値が更新されていない状態ではランダムで良い。
                return np.random.randint(len(self.actions))
            else:
                # 既知のQ値に基づき最良の行動を選択する。
                return np.argmax(self.value[state])

    def learn(
        self,
        env,
        episode_count=3000,
        gamma=0.9,
        learning_rate=0.1,
        steps_in_model=-1,
        report_interval=100,
    ):
        """
        Dyna の学習ループである。
        - 実環境で1ステップ進めて Q 値を TD(0) で更新する。
        - さらに steps_in_model > 0 の場合、内部モデルから擬似経験を
          指定回数サンプリングし、同様に Q 値を追加更新する。
        引数:
          env            : Gym 環境
          episode_count  : エピソード数
          gamma          : 割引率
          learning_rate  : 学習率（TD誤差の反映係数）
          steps_in_model : 1実ステップあたりのモデル内シミュレーション回数
          report_interval: 平均報酬の表示間隔
        """
        # 環境の離散行動空間から行動ID集合を構築する。
        self.actions = list(range(env.action_space.n))
        # Q テーブルを「未訪問は 0 の配列」で初期化する。
        self.value = defaultdict(lambda: [0] * len(self.actions))
        # 環境モデル（遷移・報酬の経験カウントによる近似）を作成する。
        model = Model(self.actions)

        rewards = []  # 各エピソードの最終報酬（FrozenLake ではクリア=1, それ以外=0）
        for e in range(episode_count):
            s = env.reset()  # エピソード開始（初期状態）
            done = False
            goal_reward = 0  # 今エピソードの最終（到達）報酬を格納

            while not done:
                # ε-greedy で行動選択
                a = self.policy(s)
                # 実環境で 1 ステップ遷移
                n_state, reward, done, info = env.step(a)

                # --- 実環境での経験に基づく TD(0) 更新 ---
                # TD ターゲット = r + γ * max_a' Q(s', a')
                gain = reward + gamma * max(self.value[n_state])
                # 現在の推定値
                estimated = self.value[s][a]
                # Q ← Q + α * (TD誤差)
                self.value[s][a] += learning_rate * (gain - estimated)

                # --- モデルベースのシミュレーション更新（Dyna 部分） ---
                if steps_in_model > 0:
                    # モデルを最新の経験で更新
                    model.update(s, a, reward, n_state)
                    # モデルから擬似遷移を一定回数サンプルして追加更新
                    for s, a, r, n_s in model.simulate(steps_in_model):
                        gain = r + gamma * max(self.value[n_s])
                        estimated = self.value[s][a]
                        self.value[s][a] += learning_rate * (gain - estimated)

                # 次状態へ
                s = n_state
            else:
                # while を done で抜けたとき、最後に得た報酬（0 or 1）を記録
                goal_reward = reward

            rewards.append(goal_reward)
            # 定期的に直近エピソードの平均報酬を表示
            if e != 0 and e % report_interval == 0:
                recent = np.array(rewards[-report_interval:])
                print("At episode {}, reward is {}".format(e, recent.mean()))


class Model:
    """
    経験に基づく環境モデルの簡易実装である。
    - transit_count[state][action]: 次状態の出現回数 Counter
    - total_reward[state][action]: 報酬の合計（期待報酬推定に使う）
    - history[state][action]     : 該当の (s,a) が観測された回数
    Dyna の擬似経験生成では、(s,a) を履歴からサンプルし、
    transit() と reward() により「次状態」と「期待報酬」を引いて返す。
    """

    def __init__(self, actions):
        self.num_actions = len(actions)
        # 遷移カウント: state -> [ Counter(next_state), ... for each action ]
        self.transit_count = defaultdict(lambda: [Counter() for a in actions])
        # 累積報酬: state -> [ total_reward_for_action, ... ]
        self.total_reward = defaultdict(lambda: [0] * self.num_actions)
        # 履歴: state -> Counter({action: count})
        self.history = defaultdict(Counter)

    def update(self, state, action, reward, next_state):
        """
        実環境で観測した (s, a, r, s') を用いてモデルを更新する。
        """
        self.transit_count[state][action][next_state] += 1
        self.total_reward[state][action] += reward
        self.history[state][action] += 1

    def transit(self, state, action):
        """
        (state, action) から次状態を確率的にサンプルする。
        観測頻度に基づく経験的確率 p(s'|s,a) で抽選する。
        """
        counter = self.transit_count[state][action]
        states = []
        counts = []
        # most_common() で（高頻度順だが）全要素を取り出す。
        for s, c in counter.most_common():
            states.append(s)
            counts.append(c)
        # 観測頻度を正規化して確率分布へ
        probs = np.array(counts) / sum(counts)
        # 観測頻度に比例した確率で次状態をサンプルする。
        return np.random.choice(states, p=probs)

    def reward(self, state, action):
        """
        (state, action) に対する期待報酬 E[r|s,a] を、
        累積報酬 / 観測回数 から推定して返す。
        """
        total_reward = self.total_reward[state][action]
        total_count = self.history[state][action]
        return total_reward / total_count

    def simulate(self, count):
        """
        モデルから擬似経験を count 回生成するジェネレータである。
        1回ごとに:
          - 履歴に存在する state をランダムサンプル
          - その state で観測回数 > 0 の action をランダムサンプル
          - transit() で次状態を、reward() で期待報酬をサンプル
          - (state, action, reward, next_state) を yield
        """
        states = list(self.transit_count.keys())

        # 当該 state において、一度でも使われた action のみ候補にする。
        actions = lambda s: [a for a, c in self.history[s].most_common() if c > 0]

        for i in range(count):
            # 擬似的に (s, a) をサンプル
            state = np.random.choice(states)
            action = np.random.choice(actions(state))

            # モデルから次状態と期待報酬を取得
            next_state = self.transit(state, action)
            reward = self.reward(state, action)

            yield state, action, reward, next_state


def main(steps_in_model):
    """
    エントリポイントである。FrozenLakeEasy を生成し、DynaAgent を学習させる。
    steps_in_model は Dyna のシミュレーション回数（1環境ステップあたり）を制御する。
    -1（デフォルト）の場合は model-based 追加更新を行わない（= Q-learning 相当の更新のみ）。
    """
    env = gym.make("FrozenLakeEasy-v0")
    agent = DynaAgent()
    agent.learn(env, steps_in_model=steps_in_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dyna Agent")
    parser.add_argument(
        "--modelstep", type=int, default=-1, help="step count in the model"
    )

    args = parser.parse_args()
    main(args.modelstep)
