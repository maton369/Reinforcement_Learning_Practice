# ============================================================
# 模倣学習（Imitation Learning）の最小実装：FrozenLake（簡易版）
# ------------------------------------------------------------
# 概要：
#   - TeacherAgent（教師）：Q学習風に MLPRegressor で Q(s,a) を近似し、ε-greedy で方策を決定する。
#   - Student（生徒）：教師の行動を模倣する分類器（MLPClassifier）。教師データを集めて学習する。
#   - FrozenLakeEasy-v0：滑らない（is_slippery=False）FrozenLake を自前登録して使用。
#
# 重要メモ：
#   - joblib の import について：新しい scikit-learn では `from joblib import dump, load`
#     もしくは `import joblib` を推奨。ここでは提示コードの構造を尊重し、コメントのみ付与。
#   - Teacher の学習は "1-step 先読み" に相当する TD 的更新（target = r + γ max_a' Q(s',a')）で、
#     MLPRegressor を毎ステップ partial_fit している（オンライン近似）。
#   - Student の学習は教師の行動ラベルを学習する模倣（Behavior Cloning）であり、
#     さらに「生徒の方策で環境を動かしつつ、教師のアクションを正解ラベルとして集める」
#     ため、DAgger（Dataset Aggregation）的な振る舞いを含む。
# ============================================================

# --- ここだけ置き換え（先頭付近の import 部分） ---
import os
import argparse
import warnings
import numpy as np

# ✅ まずは公式の joblib を使う
try:
    import joblib
except ImportError:
    # ⛳ かなり古い scikit-learn 用のフォールバック（基本は通らない想定）
    from sklearn.externals import joblib

from sklearn.neural_network import MLPRegressor, MLPClassifier
import gym
from gym.envs.registration import register

# ------------------------------------------------------------
# FrozenLake（滑らない版）を登録
#   - 既存IDと衝突する場合があるため、明示的に "FrozenLakeEasy-v0" として登録。
#   - is_slippery=False により確率的な滑りをオフにし、学習の再現性を上げる。
# ------------------------------------------------------------
register(
    id="FrozenLakeEasy-v0",
    entry_point="gym.envs.toy_text:FrozenLakeEnv",
    kwargs={"is_slippery": False},
)


class TeacherAgent:
    """
    教師エージェント：
      - MLPRegressor で Q(s,a) を近似し、ε-greedy 方策で行動を選択する。
      - 部分学習（partial_fit）を用いてオンラインで Q の近似器を更新。
      - 保存/読み込みには joblib を使用。
    """

    def __init__(self, env, epsilon=0.1):
        # 行動のインデックス集合（例：FrozenLake では 0~3）
        self.actions = list(range(env.action_space.n))
        # 探索率 ε：小さいほど貪欲、大きいほどランダム探索
        self.epsilon = epsilon
        # Q近似用モデル（後で initialize で生成）
        self.model = None

    def save(self, model_path):
        # 学習済みモデルを保存（環境復元に依存しないため実験再現に便利）
        joblib.dump(self.model, model_path)

    @classmethod
    def load(cls, env, model_path, epsilon=0.1):
        # 保存済みモデルを読み込み、教師エージェントを復元
        agent = cls(env, epsilon)
        agent.model = joblib.load(model_path)
        return agent

    def initialize(self, state):
        """
        Q(s,a) 近似器を初期化して「予測メソッドを呼べる状態」にする。
        - MLPRegressor は partial_fit の前に、出力次元を知るためのダミー学習が必要。
        - 出力は len(actions) 次元の連続値（各アクションのQ値）。
        """
        # hidden_layer_sizes=() で「線形モデル」にしている（最小構成）。
        self.model = MLPRegressor(hidden_layer_sizes=(), max_iter=1)
        # ダミーのラベル（Qベクトル）で暖機運転：これで predict が呼べるようになる。
        dummy_label = [np.random.uniform(size=len(self.actions))]
        self.model.partial_fit([state], dummy_label)
        return self

    def estimate(self, state):
        """
        近似器から Q(s, ·) を推定して返す。
        返り値：shape = (n_actions,)
        """
        q = self.model.predict([state])[0]
        return q

    def policy(self, state):
        """
        ε-greedy 方策：
          - 確率 ε でランダム行動
          - それ以外は argmax_a Q(s,a)
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.actions))
        else:
            return np.argmax(self.estimate(state))

    @classmethod
    def train(
        cls,
        env,
        episode_count=3000,
        gamma=0.9,
        initial_epsilon=1.0,
        final_epsilon=0.1,
        report_interval=100,
    ):
        """
        教師の学習ループ：
          - 1エピソードずつ、TDターゲット（r + γ max_a' Q(s',a')）で Q を更新。
          - オンライン近似：毎ステップ partial_fit。
          - ε は線形に減衰（ε-annealing）。
        """
        # 初期状態の one-hot を使って Q 近似器を初期化
        agent = cls(env, initial_epsilon).initialize(env.reset())
        rewards = []  # 各エピソードの最終報酬（FrozenLake は 0/1 が多い）
        # εを線形に減衰させるステップ幅
        decay = (initial_epsilon - final_epsilon) / episode_count

        for e in range(episode_count):
            s = (
                env.reset()
            )  # s は one-hot ベクトル（FrozenLakeObserver.transform 参照）
            done = False
            goal_reward = 0  # エピソード終了時の報酬（ゴール=1、穴=0 が多い）

            while not done:
                # 行動選択（ε-greedy）
                a = agent.policy(s)
                # 現在の Q 推定（ベクトル）
                estimated = agent.estimate(s)

                # 環境ステップ
                n_state, reward, done, info = env.step(a)

                # TD ターゲット： r + γ max_a' Q(s',a')
                gain = reward + gamma * max(agent.estimate(n_state))

                # 対応行動 a の Q(s,a) のみを上書き更新
                estimated[a] = gain

                # 1サンプルだけでオンライン学習（部分学習）
                agent.model.partial_fit([s], [estimated])

                # 次状態へ遷移
                s = n_state
            else:
                # FrozenLake は多くの場合、終了時の reward がエピソードの報酬
                goal_reward = reward

            # ログ用バッファへ積む
            rewards.append(goal_reward)

            # 学習の進捗を定期的に表示
            if e != 0 and e % report_interval == 0:
                recent = np.array(rewards[-report_interval:])
                print("At episode {}, reward is {}".format(e, recent.mean()))

            # εを徐々に減らして最終的に探索を抑える
            agent.epsilon -= decay

        return agent


class FrozenLakeObserver:
    """
    環境ラッパ（Observer）：
      - 生の離散状態（0..N-1）を one-hot ベクトルへ変換し、NN にそのまま入れられるようにする。
      - 学習コードは「状態がベクトルで来る」ことを前提にするための薄いアダプタ。
    """

    def __init__(self):
        # 指定IDで FrozenLake（滑らない版）を生成
        self._env = gym.make("FrozenLakeEasy-v0")

    @property
    def action_space(self):
        # OpenAI Gym 準拠の action_space を透過
        return self._env.action_space

    @property
    def observation_space(self):
        # 同様に観測空間を透過
        return self._env.observation_space

    def reset(self):
        # 環境の reset 結果（離散整数）を one-hot へ
        return self.transform(self._env.reset())

    def render(self):
        # FrozenLake のテキストレンダラ（盤面表示）
        self._env.render()

    def step(self, action):
        # 1ステップ進め、次状態を one-hot へ変換して返す
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        """
        離散状態（0..N-1）→ one-hot ベクトル（長さ N）。
        例：N=16, state=5 → [0,0,0,0,0,1,0,...,0]
        """
        feature = np.zeros(self.observation_space.n)
        feature[state] = 1.0
        return feature


class Student:
    """
    生徒エージェント：
      - 教師の行動を分類問題として模倣（Behavior Cloning）。
      - さらに「生徒で環境を進めつつ、各状態に対する教師の行動ラベルを収集」
        するため、DAgger 的なデータ拡張になっている。
    """

    def __init__(self, env):
        # 行動ラベルの集合（分類器のクラス集合）
        self.actions = list(range(env.action_space.n))
        # クラス分類器（後で initialize で構築）
        self.model = None

    def initialize(self, state):
        """
        分類器を初期化し、partial_fit を使えるようにするための暖機運転を行う。
        - classes に行動集合を渡して、全クラスが学習対象であることを明示する。
        """
        self.model = MLPClassifier(hidden_layer_sizes=(), max_iter=1)
        dummy_action = 0  # 適当なラベルで 1 サンプル学習
        self.model.partial_fit([state], [dummy_action], classes=self.actions)
        return self

    def policy(self, state):
        """
        現在状態に対する予測（= 行動ラベル）を返す。
        模倣学習なので argmax ではなく、分類器の predict そのものが行動。
        """
        return self.model.predict([state])[0]

    def imitate(
        self,
        env,
        teacher,
        initial_step=100,
        train_step=200,
        report_interval=10,
    ):
        """
        模倣学習の主手順：
          1) 教師方策で環境を動かしてデータ収集（states, actions）
          2) 収集データで生徒を初期学習
          3) 生徒方策で環境を動かしつつ、各状態に対する「教師の正解行動」を継続収集
          4) 新しいデータで継続的に partial_fit（=DAgger 的挙動）
        """
        states = []
        actions = []

        # 1) 教師デモの収集：教師が実際に動いた（s, a）を記録
        for e in range(initial_step):
            s = env.reset()
            done = False
            while not done:
                a = teacher.policy(s)  # 教師の方策
                n_state, reward, done, info = env.step(a)
                states.append(s)
                actions.append(a)
                s = n_state

        # 2) 生徒を初期化し、初期データで一括学習
        self.initialize(states[0])
        self.model.partial_fit(states, actions)

        print("Start imitation.")
        # 3)～4) 生徒で環境を進めつつ、教師ラベルを継続収集 → 逐次学習
        step_limit = 20  # 1エピソードの最大ステップ（安全のため打ち切り）
        for e in range(train_step):
            s = env.reset()
            done = False
            rewards = []  # FrozenLake ではエピソードの末尾報酬を記録する簡略設計
            step = 0

            while not done and step < step_limit:
                # 生徒の現在方策で行動
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)

                # その状態 s に対する教師の「正解行動」を後学習用に保存（DAgger 的）
                states.append(s)
                actions.append(teacher.policy(s))

                s = n_state
                step += 1
            else:
                goal_reward = reward  # 終了時の報酬を記録（成功=1, それ以外=0 が多い）

            rewards.append(goal_reward)

            # 進捗ログ
            if e != 0 and e % report_interval == 0:
                recent = np.array(rewards[-report_interval:])
                print("At episode {}, reward is {}".format(e, recent.mean()))

            # DeprecationWarning のノイズを抑えつつ、追加データで継続学習
            with warnings.catch_warnings():
                # 参考：https://github.com/scikit-learn/scikit-learn/issues/10449
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self.model.partial_fit(states, actions)


def main(teacher):
    """
    エントリポイント：
      - --teacher 指定時：教師（TeacherAgent）を学習 → joblib で保存
      - 未指定時：教師モデルを読み込み → 生徒（Student）に模倣学習を実施
    """
    env = FrozenLakeObserver()
    path = os.path.join(os.path.dirname(__file__), "imitation_teacher.pkl")

    if teacher:
        # 教師を学習・保存（模倣の元となるポリシーを作る）
        agent = TeacherAgent.train(env)
        agent.save(path)
    else:
        # 既存の教師を読み込んで、生徒に模倣学習させる
        teacher_agent = TeacherAgent.load(env, path)
        student = Student(env)
        student.imitate(env, teacher_agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Imitation Learning")
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="train teacher model",
    )
    args = parser.parse_args()
    main(args.teacher)
