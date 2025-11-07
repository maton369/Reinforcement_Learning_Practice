# ===============================================
# Deep Q-Network（DQN）による強化学習エージェントの実装（詳細コメント付き）
# ---------------------------------------------------------------
# ・対象環境   : PLE Catcher（またはテスト時は CartPole-v0）
# ・アルゴリズム: DQN（教師ネットワーク併用、ε-greedy 探索、経験再生バッファ）
# ・ネット構造 : Conv層3段 + 全結合層2段（Catcher）／小規模MLP（CartPole）
# ・最適化手法 : Adam（勾配クリッピング対応）
# ・特徴       : 教師ネットの周期更新・εの線形減衰・報酬最大時モデル保存
# ===============================================

import random
import argparse
from collections import deque
from types import MethodType
import numpy as np
from tensorflow import keras as K  # 公開API
from PIL import Image
import gym
import gym_ple
# from gym_ple.ple_env import PLEEnv
from fn_framework import FNAgent, Trainer, Observer  # 共通フレームワーク


# ---------------------------------------------------------------
# Deep Q-Network エージェント本体
# ---------------------------------------------------------------
class DeepQNetworkAgent(FNAgent):

    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)
        self._scaler = None
        self._teacher_model = None  # 教師ネット（ターゲットネット）

    # モデル初期化：経験バッファの状態から形状を取得しネット構築
    def initialize(self, experiences, optimizer):
        feature_shape = experiences[0].s.shape
        self.make_model(feature_shape)
        self.model.compile(optimizer, loss="mse")
        self.initialized = True
        print("Done initialization. From now, begin training!")

    # CNNモデルの構築（Catcher向け）
    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()  # Xavier初期化
        model = K.Sequential()
        # Conv層1：8x8カーネル・32ch・ストライド4
        model.add(
            K.layers.Conv2D(
                32,
                kernel_size=8,
                strides=4,
                padding="same",
                input_shape=feature_shape,
                kernel_initializer=normal,
                activation="relu",
            )
        )
        # Conv層2：4x4カーネル・64ch・ストライド2
        model.add(
            K.layers.Conv2D(
                64,
                kernel_size=4,
                strides=2,
                padding="same",
                kernel_initializer=normal,
                activation="relu",
            )
        )
        # Conv層3：3x3カーネル・64ch・ストライド1
        model.add(
            K.layers.Conv2D(
                64,
                kernel_size=3,
                strides=1,
                padding="same",
                kernel_initializer=normal,
                activation="relu",
            )
        )
        # Flatten → 全結合
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(256, kernel_initializer=normal, activation="relu"))
        # 出力層：行動数に対応するQ値ベクトル
        model.add(K.layers.Dense(len(self.actions), kernel_initializer=normal))
        self.model = model
        # 教師ネットを複製して初期化
        self._teacher_model = K.models.clone_model(self.model)

    # Q値推定（単一状態s）
    def estimate(self, state):
        return self.model.predict(np.array([state]))[0]

    # Q学習の更新処理（TDターゲットによるMSE最小化）
    def update(self, experiences, gamma):
        states = np.array([e.s for e in experiences])
        n_states = np.array([e.n_s for e in experiences])

        estimateds = self.model.predict(states)
        future = self._teacher_model.predict(n_states)

        for i, e in enumerate(experiences):
            reward = e.r
            # 終了状態でなければブートストラップ項を加える
            if not e.d:
                reward += gamma * np.max(future[i])
            estimateds[i][e.a] = reward  # TDターゲットで上書き

        # 1バッチ学習（損失を返す）
        loss = self.model.train_on_batch(states, estimateds)
        return loss

    # 教師ネット更新：オンラインネットの重みをコピー
    def update_teacher(self):
        self._teacher_model.set_weights(self.model.get_weights())


# ---------------------------------------------------------------
# テスト用DQN（軽量MLP版；CartPoleなどに使用）
# ---------------------------------------------------------------
class DeepQNetworkAgentTest(DeepQNetworkAgent):
    def __init__(self, epsilon, actions):
        super().__init__(epsilon, actions)

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(
            K.layers.Dense(
                64,
                input_shape=feature_shape,
                kernel_initializer=normal,
                activation="relu",
            )
        )
        model.add(
            K.layers.Dense(
                len(self.actions), kernel_initializer=normal, activation="relu"
            )
        )
        self.model = model
        self._teacher_model = K.models.clone_model(self.model)


# ---------------------------------------------------------------
# 環境観測の前処理（Catcher専用）
# ---------------------------------------------------------------
class CatcherObserver(Observer):
    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)  # フレームスタック

    # 状態を (height, width, frame_count) のテンソルに変換
    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")  # グレースケール化
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # 正規化 (0〜1)
        # 初期時は同一フレームをframe_count回挿入
        if len(self._frames) == 0:
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # 軸を (frame, width, height) → (height, width, frame) に転置
        feature = np.transpose(feature, (1, 2, 0))
        return feature


# ---------------------------------------------------------------
# DQNトレーナー：学習ループとハイパーパラメータ制御
# ---------------------------------------------------------------
class DeepQNetworkTrainer(Trainer):
    def __init__(
        self,
        buffer_size=50000,
        batch_size=32,
        gamma=0.99,
        initial_epsilon=0.5,
        final_epsilon=1e-3,
        learning_rate=1e-3,
        teacher_update_freq=3,
        report_interval=10,
        log_dir="",
        file_name="",
    ):
        super().__init__(buffer_size, batch_size, gamma, report_interval, log_dir)
        self.file_name = file_name if file_name else "dqn_agent.h5"
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.learning_rate = learning_rate
        self.teacher_update_freq = teacher_update_freq
        self.loss = 0
        self.training_episode = 0
        self._max_reward = -10  # モデル保存用の閾値

    # 学習実行：訓練モード／テストモードで分岐
    def train(
        self,
        env,
        episode_count=1200,
        initial_count=200,
        test_mode=False,
        render=False,
        observe_interval=100,
    ):
        actions = list(range(env.action_space.n))
        if not test_mode:
            agent = DeepQNetworkAgent(1.0, actions)
        else:
            agent = DeepQNetworkAgentTest(1.0, actions)
            observe_interval = 0
        self.training_episode = episode_count

        self.train_loop(
            env, agent, episode_count, initial_count, render, observe_interval
        )
        return agent

    # 各エピソード開始時に損失を初期化
    def episode_begin(self, episode, agent):
        self.loss = 0

    # 初期化時にネット構築・コンパイル・TensorBoardセットアップ
    def begin_train(self, episode, agent):
        optimizer = K.optimizers.Adam(learning_rate=self.learning_rate, clipvalue=1.0)
        agent.initialize(self.experiences, optimizer)
        self.logger.set_model(agent.model)
        agent.epsilon = self.initial_epsilon
        self.training_episode -= episode

    # 各ステップで経験をサンプリングし1バッチ更新
    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            self.loss += agent.update(batch, self.gamma)

    # 各エピソード終了時の処理
    def episode_end(self, episode, step_count, agent):
        reward = sum([e.r for e in self.get_recent(step_count)])  # 報酬合計
        self.loss = self.loss / step_count
        self.reward_log.append(reward)

        if self.training:
            # TensorBoardへログ書き込み
            self.logger.write(self.training_count, "loss", self.loss)
            self.logger.write(self.training_count, "reward", reward)
            self.logger.write(self.training_count, "epsilon", agent.epsilon)
            # ベスト報酬でモデル保存
            if reward > self._max_reward:
                agent.save(self.logger.path_of(self.file_name))
                self._max_reward = reward
            # 教師ネット更新
            if self.is_event(self.training_count, self.teacher_update_freq):
                agent.update_teacher()
            # εを線形減衰
            diff = self.initial_epsilon - self.final_epsilon
            decay = diff / self.training_episode
            agent.epsilon = max(agent.epsilon - decay, self.final_epsilon)

        # ログ出力
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval :]
            self.logger.describe("reward", recent_rewards, episode=episode)


# ---------------------------------------------------------------
# メイン関数：学習モード or プレイモード
# ---------------------------------------------------------------
def main(play, is_test):
    file_name = "dqn_agent.h5" if not is_test else "dqn_agent_test.h5"
    trainer = DeepQNetworkTrainer(file_name=file_name)
    path = trainer.logger.path_of(trainer.file_name)
    agent_class = DeepQNetworkAgent

    if is_test:
        print("Train on test mode")
        obs = gym.make("CartPole-v0")
        agent_class = DeepQNetworkAgentTest
    else:
        # ★ 修正：PLE から直接 Catcher 環境を生成
        env = gym.make("Catcher-v0")
        _ensure_ple_compat(env)
        obs = CatcherObserver(env, 80, 80, 4)
        trainer.learning_rate = 1e-4

    if play:
        agent = agent_class.load(obs, path)
        agent.play(obs, render=True)
    else:
        trainer.train(obs, test_mode=is_test)


def _ensure_ple_compat(env):
    """
    gym-ple(PLEEnv) は旧APIの _reset/_step/_render/_seed を実装している。
    Modern Gym からの呼び出しに耐えるよう、余分な引数を吸収するシムを当てる。
    """
    from types import MethodType

    base = getattr(env, "unwrapped", env)

    # reset: Gymは reset(seed=..., options=...) を渡す可能性があるため **kwargs を吸収
    if hasattr(base, "_reset"):

        def reset_shim(self, **kwargs):
            # PLEの _reset は引数なし想定
            return self._reset()

        base.reset = MethodType(reset_shim, base)

    # step: 旧APIは (action) のみ。戻りも旧API互換でOK（fn_framework側で吸収済み）
    if hasattr(base, "_step"):

        def step_shim(self, action, *args, **kwargs):
            return self._step(action)

        base.step = MethodType(step_shim, base)

    # render: Gymは mode="human" 等を渡すのでデフォルト引数を用意
    if hasattr(base, "_render"):

        def render_shim(self, mode="human", *args, **kwargs):
            return self._render(mode)

        base.render = MethodType(render_shim, base)

    # seed: Gymは seed(None) などを渡すのでデフォルト付きで受ける
    if hasattr(base, "_seed"):

        def seed_shim(self, seed=None, *args, **kwargs):
            return self._seed(seed)

        base.seed = MethodType(seed_shim, base)

# ---------------------------------------------------------------
# CLI引数設定：--play でプレイモード、--test で簡易訓練モード
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")
    parser.add_argument("--test", action="store_true", help="train by test mode")

    args = parser.parse_args()
    main(args.play, args.test)
