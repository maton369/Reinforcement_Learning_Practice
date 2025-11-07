# ===============================================
# Advantage Actor-Critic（A2C系）エージェント実装（詳細コメント付き）
# 目的：
#   - 画像入力（Catcher）/ 低次元入力（CartPoleテスト）に対して、Actor（方策）とCritic（価値）を同時学習する。
#   - 損失は policy_loss + value_loss_weight * value_loss - entropy_weight * entropy で最適化する。
# 特徴：
#   - SampleLayer により Gumbel-Max trick を用いてロジットから1アクションをサンプリングする。
#   - TF2環境でも v1 互換のグラフ実行（tf.compat.v1.disable_eager_execution）で古いKerasの書法を維持する。
# 注意：
#   - keras最適化器の get_updates を利用するため、TF/Keras バージョンの互換性に留意が必要である。
#   - Gym APIの仕様差異（reset/stepの戻り値など）には別途ラッパ側で対応することが望ましい。
# ===============================================

import argparse
from collections import deque
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.python import (
    keras as K,
)  # Keras（TFバンドル版）内部APIである（互換維持のため）
from tensorflow.compat.v1.keras import optimizers as v1_optimizers
from PIL import Image
import gym
import gym_ple  # PLEベースの環境群（Catcherを使う）
from fn_framework import (
    FNAgent,
    Trainer,
    Observer,
)  # 共有フレームワーク：Agent/Trainer/Observerの基底

tf.compat.v1.disable_eager_execution()  # TF2でもv1グラフ実行モードへ切替（backend.function等を使うため）


# ---------------------------------------------------------------
# Actor-Critic 本体
#   - Actor：状態→行動ロジット（評価値）を出力
#   - Critic：状態→状態価値 V(s) を出力
#   - サンプルは SampleLayer（Gumbel-Max）で1アクションを引く
# ---------------------------------------------------------------
class ActorCriticAgent(FNAgent):

    def __init__(self, actions):
        # ActorCriticAgent は自己方策（εを使わない）である
        super().__init__(epsilon=0.0, actions=actions)
        self._updater = None  # Keras backend.function による更新関数を保持する

    @classmethod
    def load(cls, env, model_path):
        # 保存済みモデルをロードして推論可能なエージェントを復元する
        actions = list(range(env.action_space.n))  # 行動空間サイズを環境から決定
        agent = cls(actions)
        # SampleLayer を custom_objects に渡してモデルを復元
        agent.model = K.models.load_model(
            model_path, custom_objects={"SampleLayer": SampleLayer}
        )
        agent.initialized = True
        return agent

    def initialize(self, experiences, optimizer):
        # 収集済み経験から入力形状を取得してネットワークを構築し、更新関数を用意する
        feature_shape = experiences[0].s.shape  # 画像入力なら (H,W,C)
        self.make_model(feature_shape)  # Actor/Criticの双頭モデルを生成
        self.set_updater(optimizer)  # 損失定義と更新opを構築
        # v1 グラフ実行では変数初期化を明示的に行う
        sess = K.backend.get_session()
        sess.run(tf.compat.v1.global_variables_initializer())
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def make_model(self, feature_shape):
        # 画像入力前提のCNNベースの特徴抽出器→Actor/Criticヘッド
        normal = K.initializers.glorot_normal()  # Xavier正規初期化
        model = K.Sequential()
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
        model.add(K.layers.Flatten())
        model.add(K.layers.Dense(256, kernel_initializer=normal, activation="relu"))

        # --- Actor ヘッド：各行動のロジット（action_evals）を出力 ---
        actor_layer = K.layers.Dense(len(self.actions), kernel_initializer=normal)
        action_evals = actor_layer(model.output)  # shape: (B, n_actions)
        actions = SampleLayer()(
            action_evals
        )  # Gumbel-Maxで1アクションを整数IDとしてサンプル

        # --- Critic ヘッド：状態価値 V(s) を出力 ---
        critic_layer = K.layers.Dense(1, kernel_initializer=normal)
        values = critic_layer(model.output)  # shape: (B, 1)

        # モデル入出力をまとめた K.Model として保持
        self.model = K.Model(
            inputs=model.input, outputs=[actions, action_evals, values]
        )

    def set_updater(self, optimizer, value_loss_weight=1.0, entropy_weight=0.1):
        # A2Cの損失を定義し、Keras backend.function で更新関数を作る
        actions = tf.compat.v1.placeholder(
            shape=(None), dtype="int32"
        )  # サンプル済み「実行行動」a_t
        values = tf.compat.v1.placeholder(
            shape=(None), dtype="float32"
        )  # 教師となるターゲット価値（割引和など）

        # モデル出力：actions（argmaxサンプル）, action_evals（ロジット）, estimateds（価値V）
        _, action_evals, estimateds = self.model.output

        # --- Policy（Actor）損失 ---
        # neg_logs = - log π(a_t | s_t) を sparse_softmax_xent で計算する（内部でsoftmax + logを計算）
        neg_logs = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=action_evals, labels=actions
        )

        # Advantage = (ターゲット値) - stop_gradient(V(s)) とする
        # stop_gradient により Actor の更新が Critic へ逆流しないようにする
        advantages = values - tf.stop_gradient(estimateds)

        policy_loss = tf.reduce_mean(neg_logs * advantages)  # E[ -logπ * A ]

        # --- Value（Critic）損失 ---
        value_loss = tf.keras.losses.MeanSquaredError()(
            values, estimateds
        )  # (ターゲット − V(s))^2

        # --- エントロピー正則化（方策の多様性を促進）---
        action_entropy = tf.reduce_mean(self.categorical_entropy(action_evals))

        # --- 総合損失 ---
        loss = policy_loss + value_loss_weight * value_loss
        loss -= (
            entropy_weight * action_entropy
        )  # エントロピーを最大化（=損失へは負符号）

        # Keras Optimizer（v1流）から更新opを取得
        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)

        # backend.function で「順伝播→損失/各種メトリクス出力＋重み更新」をひとまとめにする
        self._updater = K.backend.function(
            inputs=[self.model.input, actions, values],
            outputs=[
                loss,  # 総損失
                policy_loss,  # 方策損失
                value_loss,  # 価値損失
                tf.reduce_mean(neg_logs),  # 平均 -logπ
                tf.reduce_mean(advantages),  # 平均 A
                action_entropy,
            ],  # 平均エントロピー
            updates=updates,
        )

    def categorical_entropy(self, logits):
        """
        softmax分布のエントロピーをロジットから計算する。
        OpenAI Baselines 実装に準拠：
        https://github.com/openai/baselines/blob/master/baselines/common/distributions.py#L192
        """
        a0 = logits - tf.reduce_max(
            logits, axis=-1, keepdims=True
        )  # 数値安定化のため max を引く
        ea0 = tf.exp(a0)
        z0 = tf.reduce_sum(ea0, axis=-1, keepdims=True)
        p0 = ea0 / z0
        return tf.reduce_sum(p0 * (tf.math.log(z0) - a0), axis=-1)  # = -Σ p log p

    def policy(self, s):
        # 現在方策から1アクションをサンプルして返す（未初期化ならランダム）
        if not self.initialized:
            return np.random.randint(len(self.actions))
        else:
            action, action_evals, values = self.model.predict(np.array([s]))
            return action[0]  # SampleLayerのargmax出力（整数ID）

    def estimate(self, s):
        # 現在の Critic による状態価値推定 V(s) を返す（スカラー）
        action, action_evals, values = self.model.predict(np.array([s]))
        return values[0][0]

    def update(self, states, actions, rewards):
        # 1 バッチ分の更新を実行する（rewards はターゲット価値に相当）
        return self._updater([states, actions, rewards])


# ---------------------------------------------------------------
# カスタム層：Gumbel-Max trick によりロジットから1アクションをサンプル
#   argmax(logits + Gumbel(0,1)) と同値な x - log(-log(U)) により実装
# ---------------------------------------------------------------
class SampleLayer(K.layers.Layer):

    def __init__(self, **kwargs):
        self.output_dim = 1  # 1ステップにつき1アクションを返す
        super(SampleLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 学習パラメータは特に持たないため何もしない
        super(SampleLayer, self).build(input_shape)

    def call(self, x):
        # Gumbel-Max: argmax(logits - log(-log(U)))
        noise = tf.random.uniform(tf.shape(x))
        return tf.argmax(x - tf.math.log(-tf.math.log(noise)), axis=1)

    def compute_output_shape(self, input_shape):
        # 出力は (B, 1) 形式（Bはバッチサイズ）
        return (input_shape[0], self.output_dim)


# ---------------------------------------------------------------
# テスト用の軽量Actor-Critic（MLP版；CartPole向け）
#   - 特徴抽出をMLPに簡略化した以外は上記と同様の構成
# ---------------------------------------------------------------
class ActorCriticAgentTest(ActorCriticAgent):

    def make_model(self, feature_shape):
        normal = K.initializers.glorot_normal()
        model = K.Sequential()
        model.add(
            K.layers.Dense(
                10,
                input_shape=feature_shape,
                kernel_initializer=normal,
                activation="relu",
            )
        )
        model.add(K.layers.Dense(10, kernel_initializer=normal, activation="relu"))

        actor_layer = K.layers.Dense(len(self.actions), kernel_initializer=normal)

        action_evals = actor_layer(model.output)  # ロジット
        actions = SampleLayer()(action_evals)  # Gumbel-Maxサンプル

        critic_layer = K.layers.Dense(1, kernel_initializer=normal)
        values = critic_layer(model.output)  # V(s)

        self.model = K.Model(
            inputs=model.input, outputs=[actions, action_evals, values]
        )


# ---------------------------------------------------------------
# 画像観測の前処理ラッパ（Catcher用）
#   - グレースケール化→リサイズ→[0,1]正規化→フレームスタック→(H,W,F) へ転置
# ---------------------------------------------------------------
class CatcherObserver(Observer):

    def __init__(self, env, width, height, frame_count):
        super().__init__(env)
        self.width = width
        self.height = height
        self.frame_count = frame_count
        self._frames = deque(maxlen=frame_count)  # 最近のフレームをスタック

    def transform(self, state):
        grayed = Image.fromarray(state).convert("L")  # グレースケール
        resized = grayed.resize((self.width, self.height))
        resized = np.array(resized).astype("float")
        normalized = resized / 255.0  # [0,1]へスケーリング
        if len(self._frames) == 0:
            # 初回は同一フレームを frame_count 回入れてスタックを満たす
            for i in range(self.frame_count):
                self._frames.append(normalized)
        else:
            self._frames.append(normalized)
        feature = np.array(self._frames)
        # 形状を (F, W, H) → (H, W, F) へ並べ替え（CNNのチャネル次元として積む）
        feature = np.transpose(feature, (1, 2, 0))
        return feature


# ---------------------------------------------------------------
# 学習の司令塔：A2Cのバッチ生成・更新・ログ記録・モデル保存など
# ---------------------------------------------------------------
class ActorCriticTrainer(Trainer):

    def __init__(
        self,
        buffer_size=256,
        batch_size=32,
        gamma=0.99,
        learning_rate=1e-3,
        report_interval=10,
        log_dir="",
        file_name="",
    ):
        # buffer_size: 初期化前に溜める最小経験数
        # batch_size : 1回の更新に使うサンプル数
        # gamma      : 割引率
        # learning_rate: 最適化の学習率
        # report_interval: 何エピソードごとに統計を表示するか
        super().__init__(buffer_size, batch_size, gamma, report_interval, log_dir)
        self.file_name = file_name if file_name else "a2c_agent.h5"
        self.learning_rate = learning_rate
        self.losses = {}  # 直近の損失・メトリクスを一時保持してロギングする
        self.rewards = []  # エピソード内の逐次報酬を保存する
        self._max_reward = -10  # ベストモデル保存の閾値（初期は小さめに設定）

    def train(
        self,
        env,
        episode_count=900,
        initial_count=10,
        test_mode=False,
        render=False,
        observe_interval=100,
    ):
        # 環境の行動空間からアクションの集合を作成し、学習/テストでAgent種別を切替える
        actions = list(range(env.action_space.n))
        if not test_mode:
            agent = ActorCriticAgent(actions)
        else:
            agent = ActorCriticAgentTest(actions)
            observe_interval = 0  # テスト時は観測画像の保存等を抑制する想定
        self.training_episode = episode_count

        # 共通ループを実行（環境反復、経験蓄積、更新などは基底Trainerに依存）
        self.train_loop(
            env, agent, episode_count, initial_count, render, observe_interval
        )
        return agent

    def episode_begin(self, episode, agent):
        # エピソード開始ごとに逐次報酬の蓄積をクリア
        self.rewards = []

    def step(self, episode, step_count, agent, experience):
        # 1ステップの経験を受け取り、初期化前/後で分岐して処理する
        self.rewards.append(experience.r)

        if not agent.initialized:
            # まずは buffer_size まで経験を貯める（初期ネット構築に必要）
            if len(self.experiences) < self.buffer_size:
                # まだ足りないので更新は行わず return False（継続）
                return False

            # 初期化条件を満たしたらオプティマイザを用意して初期化を実行
            optimizer = v1_optimizers.Adam(
                lr=self.learning_rate, clipnorm=5.0
            )  # 勾配ノルムクリップで安定化
            agent.initialize(self.experiences, optimizer)
            self.logger.set_model(agent.model)  # ロガーへモデルを紐づけ
            self.training = True
            self.experiences.clear()  # 初期化後は一度バッファをクリア
        else:
            # すでに初期化済：更新用に十分なミニバッチが貯まっているか確認
            if len(self.experiences) < self.batch_size:
                return False

            # バッチを作成して1回の更新を実行
            batch = self.make_batch(agent)
            loss, lp, lv, p_ng, p_ad, p_en = agent.update(*batch)

            # 直近の各種メトリクスを記録（logger.writeはepisode_endで出力）
            self.losses["loss/total"] = loss
            self.losses["loss/policy"] = lp
            self.losses["loss/value"] = lv
            self.losses["policy/neg_logs"] = p_ng
            self.losses["policy/advantage"] = p_ad
            self.losses["policy/entropy"] = p_en
            self.experiences.clear()  # 使い切ったらバッファをクリア

    def make_batch(self, agent):
        # 現在バッファに溜まっている連続経験から、ターゲット価値（割引和）を計算してバッチを返す
        states = []
        actions = []
        values = []
        experiences = list(self.experiences)  # シャローコピーして順序を維持
        states = np.array([e.s for e in experiences])
        actions = np.array([e.a for e in experiences])

        # ターゲット値（bootstrapping付きの割引和）を逆順に計算
        # 最後のサンプルが終端でない場合は V(next_state) を将来価値 future として用いる
        last = experiences[-1]
        future = last.r if last.d else agent.estimate(last.n_s)
        for e in reversed(experiences):
            value = e.r
            if not e.d:
                value += self.gamma * future
            values.append(value)
            future = value
        values = np.array(list(reversed(values)))  # 計算順を元に戻す

        # 値ターゲットのスケールをそろえる（標準化）ことで学習を安定化
        scaler = StandardScaler()
        values = scaler.fit_transform(values.reshape((-1, 1))).flatten()

        return states, actions, values

    def episode_end(self, episode, step_count, agent):
        # エピソード終了時：報酬をロギングし、ベスト更新ならモデル保存
        reward = sum(self.rewards)
        self.reward_log.append(reward)

        if agent.initialized:
            # 学習が始まっている場合のみ詳細メトリクスを記録
            self.logger.write(self.training_count, "reward", reward)
            self.logger.write(self.training_count, "reward_max", max(self.rewards))

            for k in self.losses:
                self.logger.write(self.training_count, k, self.losses[k])

            # ベスト報酬を更新したらモデルを保存
            if reward > self._max_reward:
                agent.save(self.logger.path_of(self.file_name))
                self._max_reward = reward

        # 規定間隔で最近の報酬統計を表示
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval :]
            self.logger.describe("reward", recent_rewards, episode=episode)


def _ensure_ple_compat(env):
    """
    gym-ple(PLEEnv) は旧APIの _reset/_step/_render/_seed を実装している。
    Modern Gym からの呼び出しに耐えるよう、必要に応じてシムを当てる。
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

# ---------------------------------------------------------------
# エントリポイント：学習/プレイの分岐、環境構築、ハイパーパラメータ設定
# ---------------------------------------------------------------
def main(play, is_test):
    file_name = "a2c_agent.h5" if not is_test else "a2c_agent_test.h5"
    trainer = ActorCriticTrainer(file_name=file_name)
    path = trainer.logger.path_of(trainer.file_name)
    agent_class = ActorCriticAgent

    if is_test:
        # テストモード：低次元の CartPole を使用し、軽量MLP版エージェントに切替
        print("Train on test mode")
        obs = gym.make("CartPole-v0")
        agent_class = ActorCriticAgentTest
    else:
        # 学習モード：Catcher（PLE）を利用。観測は画像→フレームスタックの前処理を適用
        env = gym.make("Catcher-v0")
        _ensure_ple_compat(env)                 # ★ 互換シムを適用（重要）
        obs = CatcherObserver(env, 80, 80, 4)
        trainer.learning_rate = 7e-5            # 画像入力では学習率を小さめに設定して安定化

    if play:
        # 保存済みモデルでプレイ（評価）する
        agent = agent_class.load(obs, path)
        agent.play(obs, episode_count=10, render=True)
    else:
        # 学習を実行
        trainer.train(obs, test_mode=is_test)


# ---------------------------------------------------------------
# CLI引数処理：--play でプレイ、--test でテスト（軽量）モード学習
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A2C Agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")
    parser.add_argument("--test", action="store_true", help="train by test mode")

    args = parser.parse_args()
    main(args.play, args.test)
