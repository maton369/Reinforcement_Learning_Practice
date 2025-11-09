# ============================================================
# 進化戦略（Evolution Strategies; ES/NES系）による方策探索エージェント
# ------------------------------------------------------------
# ・目的   : Catcher（gym_ple）を対象に、ニューラルネットの重みをガウス摂動で
#            探索し、集団（population）の平均的な報酬勾配推定から重みを更新する。
# ・方法   : パラメータ空間探索（Parameter-space exploration）
#            - 各世代でベース重みに N(0, I) ノイズを加えた個体を並列評価
#            - 報酬でノイズを重み付けて、期待勾配 E[R * noise] を近似し更新
# ・備考   : 学習（update）は微分不要（black-box最適化）。Kerasのforwardのみ使用。
#            sklearn.externals.joblib は新環境では廃止のため、joblib を直接使用する。
#            gym-ple 環境は旧Gym APIのため、互換シムを適用して reset/step 等を受ける。
# ============================================================

import os
import argparse
import numpy as np

# --- 修正点１：joblib の直接インポート（scikit-learn 依存を排除） ---
try:
    from joblib import Parallel, delayed
except Exception:
    # どうしても古い環境を想定するならフォールバック（推奨はしない）
    from sklearn.externals.joblib import Parallel, delayed  # type: ignore

from PIL import Image
import matplotlib.pyplot as plt
import gym

# ------------------------------------------------------------
# 並列実行時のCUDA無効化（GPU初期化衝突を避けるための安全策）
#   - Windows: "-1" で全GPUを見せない
#   - *nix系 : ""   で空にして抑制（環境差による）
# また、TFのログを抑制する（"3": ERROR以上のみ）
# ------------------------------------------------------------
if os.name == "nt":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from tensorflow.python import keras as K  # バンドル版Keras（内部API）。互換性に注意。


# ------------------------------------------------------------
# 修正点２：gym-ple（旧API）を Modern Gym で動かすための互換シム
# ------------------------------------------------------------
def _ensure_ple_compat(env):
    """
    gym-ple(PLEEnv) は旧APIの _reset/_step/_render/_seed を実装している。
    Modern Gym からの呼び出しに耐えるよう、必要に応じてパブリックメソッドを差し替える。
    """
    from types import MethodType

    base = getattr(env, "unwrapped", env)

    def _needs_patch(public_name: str) -> bool:
        public_impl = getattr(base.__class__, public_name, None)
        return public_impl is None or public_impl == getattr(gym.Env, public_name, None)

    if hasattr(base, "_reset") and _needs_patch("reset"):

        def reset_shim(self, **kwargs):
            return self._reset()

        base.reset = MethodType(reset_shim, base)

    if hasattr(base, "_step") and _needs_patch("step"):

        def step_shim(self, action, *args, **kwargs):
            return self._step(action)

        base.step = MethodType(step_shim, base)

    if hasattr(base, "_render") and _needs_patch("render"):

        def render_shim(self, mode="human", *args, **kwargs):
            return self._render(mode)

        base.render = MethodType(render_shim, base)

    if hasattr(base, "_seed") and _needs_patch("seed"):

        def seed_shim(self, seed=None, *args, **kwargs):
            return self._seed(seed)

        base.seed = MethodType(seed_shim, base)


class EvolutionalAgent:
    """
    進化戦略により学習された方策ネット（Conv→Flatten→Dense(softmax)）で
    行動を確率的にサンプルするエージェント。

    actions: 行動インデックスのリスト（例: [0,1,2]）
    model  : Kerasモデル（重みは外部から付与/保存/読込）
    """

    def __init__(self, actions):
        self.actions = actions
        self.model = None  # Kerasの順伝播のみ使う（学習はTrainer側で実施）

    def save(self, model_path):
        # Optimizerは含めず（include_optimizer=False）、純粋な重み・構造のみ保存
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path):
        # 保存済みモデルをロードして、環境から行動数を取得して初期化
        actions = list(range(env.action_space.n))
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)
        return agent

    def initialize(self, state, weights=()):
        """
        ベースネットワークを構築。
        - Conv2D(3, k=5, s=3) → Flatten → Dense(n_actions, softmax)
        - 入力形状は観測の shape に合わせる（例: (H,W,1)）
        weights: 既存の重み（集団評価でサンプリングした個体の重み）を受け取る場合に設定
        """
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(
            K.layers.Conv2D(
                3,
                kernel_size=5,
                strides=3,
                input_shape=state.shape,
                kernel_initializer=normal,
                activation="relu",
            )
        )
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(len(self.actions), activation="softmax"))
        self.model = model
        if len(weights) > 0:
            # 個体評価用に外から渡された重みセットへ置き換える
            self.model.set_weights(weights)

    def policy(self, state):
        """
        現在状態から行動確率π(a|s)を出し、その分布に従って1アクションをサンプル。
        ESは微分を使わないため、ここは推論のみ。
        """
        action_probs = self.model.predict(np.array([state]))[0]
        action = np.random.choice(self.actions, size=1, p=action_probs)[0]
        return action

    def play(self, env, episode_count=5, render=True):
        """
        学習済み（または指定重みの）方策でテストプレイ。
        各エピソードの総報酬を標準出力へ出す。
        """
        for e in range(episode_count):
            s = env.reset()
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = env.step(a)
                episode_reward += reward
                s = n_state
            else:
                print("Get reward {}".format(episode_reward))


class CatcherObserver:
    """
    画像観測の前処理を行うラッパ。
    - gym_ple の "Catcher-v0" を内部で生成
    - グレースケール化 → リサイズ → [0,1]正規化 → (H,W,1)へ拡張
    ESは入力の前処理を軽めにして、方策ネットのサイズも小さめ（Conv 1層）に抑える。
    """

    def __init__(self, width, height, frame_count):
        import gym_ple  # ここでimportしておく（依存をこのクラス内に閉じる意図）

        self._env = gym.make("Catcher-v0")
        _ensure_ple_compat(self._env)  # ★ 互換シムを適用（重要）
        self.width = width
        self.height = height
        # frame_countは現状未使用（将来的にフレームスタックするなら活用）

    @property
    def action_space(self):
        return self._env.action_space

    @property
    def observation_space(self):
        return self._env.observation_space

    def reset(self):
        # 環境の初期化観測に transform を適用して返す
        return self.transform(self._env.reset())

    def render(self):
        # 人間向けウィンドウ描画
        self._env.render(mode="human")

    def step(self, action):
        # 環境1ステップ進め、観測に transform を適用して返す
        n_state, reward, done, info = self._env.step(action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        """
        画像を:
          - グレースケール化
          - 指定サイズへリサイズ
          - [0,1] 正規化
          - (H,W,1) チャネル次元を追加
        という形へ整形して、CNNへ入力しやすいテンソルにする。
        """
        grayed = Image.fromarray(state).convert("L")
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # scale to 0~1
        normalized = np.expand_dims(normalized, axis=2)  # H x W => H x W x C(=1)
        return normalized


class EvolutionalTrainer:
    """
    進化戦略トレーナ:
      - 各世代で population_size 個の個体を並列評価（joblib）
      - ベース重みにガウスノイズを加えた個体の報酬を用いて、期待勾配を近似
      - learning_rate / (N * sigma) * Σ_i r_i * noise_i で重みを更新
    """

    def __init__(
        self, population_size=20, sigma=0.5, learning_rate=0.1, report_interval=10
    ):
        self.population_size = population_size  # 集団（個体）数
        self.sigma = sigma  # ノイズの標準偏差
        self.learning_rate = learning_rate  # 更新スケール
        self.weights = ()  # ベース重み（反復で更新）
        self.reward_log = []  # 各世代での個体報酬（配列）を貯める
        # report_interval は標準出力ログ用（メソッド log 内で使用）

    def train(self, epoch=100, episode_per_agent=1, render=False):
        """
        学習エントリ:
          1) 環境と初期エージェントを用意してベース重みを取得
          2) 各世代で population_size の個体を並列評価（run_agent）
          3) 結果から重みを update → ログ
        戻り値は最終世代のベース重みを持つエージェント
        """
        env = self.make_env()
        actions = list(range(env.action_space.n))
        s = env.reset()
        agent = EvolutionalAgent(actions)
        agent.initialize(s)
        self.weights = agent.model.get_weights()  # ベース重み

        # joblib.Parallel によりCPU並列で個体を評価
        with Parallel(n_jobs=-1) as parallel:
            for e in range(epoch):
                experiment = delayed(EvolutionalTrainer.run_agent)
                results = parallel(
                    experiment(episode_per_agent, self.weights, self.sigma)
                    for p in range(self.population_size)
                )
                self.update(results)  # 報酬に比例する方向へ重みを押し上げる
                self.log()  # 世代ごとの統計を出力

        # 学習済み重みをセットして返す
        agent.model.set_weights(self.weights)
        return agent

    @classmethod
    def make_env(cls):
        # 評価用のCatcher環境（前処理付き）を生成
        return CatcherObserver(width=50, height=50, frame_count=5)

    @classmethod
    def run_agent(cls, episode_per_agent, base_weights, sigma, max_step=1000):
        """
        一個体の評価:
          - ベース重みにガウスノイズを加えた重み new_weights を生成
          - episode_per_agent 本のエピソードを実行し平均報酬を返す
          - あわせて使用したノイズ（各重みテンソルと同形）を返す
        """
        env = cls.make_env()
        actions = list(range(env.action_space.n))
        agent = EvolutionalAgent(actions)

        noises = []  # 各重みテンソルに対応するノイズを保持
        new_weights = []  # 摂動後の重み

        # --- 個体の重みを作る（w_i + sigma * noise_i）---
        for w in base_weights:
            noise = np.random.randn(*w.shape)
            new_weights.append(w + sigma * noise)
            noises.append(noise)

        # --- 個体の報酬を評価 ---
        total_reward = 0
        for e in range(episode_per_agent):
            s = env.reset()
            if agent.model is None:
                # 最初のタイミングでネットワークを構築し、個体の重みを適用
                agent.initialize(s, new_weights)
            done = False
            step = 0
            while not done and step < max_step:
                a = agent.policy(s)
                n_state, reward, done, info = env.step(a)
                total_reward += reward
                s = n_state
                step += 1

        reward = total_reward / episode_per_agent
        return reward, noises  # 学習側は（報酬, ノイズ列）を使って勾配を近似

    def update(self, agent_results):
        """
        個体の（報酬, ノイズ）から、ESの推定勾配でベース重みを更新。
        - 標準化報酬 normalized_rs を用いてスケールの影響を抑える
        - 各重みテンソル i ごとに Σ_j normalized_r_j * noise_{j,i} を集約
        """
        rewards = np.array([r[0] for r in agent_results])  # shape: (P,)
        noises = np.array(
            [r[1] for r in agent_results]
        )  # list(P) of list(L) of np.ndarray

        # 分散ゼロ対策（全個体同一報酬など）: std=0 の場合は更新をスキップ
        std = rewards.std()
        if std == 0:
            self.reward_log.append(rewards)
            return

        normalized_rs = (rewards - rewards.mean()) / std

        # --- ベース重みを更新 ---
        new_weights = []
        for i, w in enumerate(self.weights):
            # i番目の重みに対応する全個体のノイズ行列をまとめる
            noise_at_i = np.array([n[i] for n in noises])  # shape: (P, *w.shape)
            # 学習率の規格化（OpenAI-ESやNESでよく用いられる形）
            rate = self.learning_rate / (self.population_size * self.sigma)
            # <noise, reward> の重み付き和（次元を合わせて加算）
            w = w + rate * np.dot(noise_at_i.T, normalized_rs).T
            new_weights.append(w)

        self.weights = new_weights
        self.reward_log.append(rewards)

    def log(self):
        """
        最新世代の個体報酬統計を出力（平均/最大/最小）。
        report_intervalは設計上あるが、ここでは毎世代出している。
        """
        rewards = self.reward_log[-1]
        print(
            "Epoch {}: reward {:.3f} (max:{:.3f}, min:{:.3f})".format(
                len(self.reward_log), rewards.mean(), rewards.max(), rewards.min()
            )
        )

    def plot_rewards(self):
        """
        学習曲線（各世代での個体報酬の平均と標準偏差帯）をMatplotlibで描画。
        研究/発表用の可視化に使う。
        """
        indices = range(len(self.reward_log))
        means = np.array([rs.mean() for rs in self.reward_log])
        stds = np.array([rs.std() for rs in self.reward_log])
        plt.figure()
        plt.title("Reward History")
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
        plt.plot(indices, means, "o-", color="g", label="reward")
        plt.legend(loc="best")
        plt.show()


def main(play):
    """
    - --play なし: ESで学習→モデル保存→学習曲線プロット
    - --play あり: 保存済みモデルを読み込み、可視化つきで数エピソード実行
    """
    model_path = os.path.join(os.path.dirname(__file__), "ev_agent.h5")

    if play:
        env = EvolutionalTrainer.make_env()
        agent = EvolutionalAgent.load(env, model_path)
        agent.play(env, episode_count=5, render=True)
    else:
        trainer = EvolutionalTrainer()
        trained = trainer.train()
        trained.save(model_path)
        trainer.plot_rewards()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evolutional Agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")

    args = parser.parse_args()
    main(args.play)
