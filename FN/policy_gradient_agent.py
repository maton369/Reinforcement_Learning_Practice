# ===============================================
# Policy Gradient（REINFORCE）による方策勾配エージェント一式
# 目的：
#   - CartPole-v0 を対象として、方策π(a|s;θ)を直接最適化する学習コードである。
#   - ニューラルネットは状態→各行動の確率（softmax）を出力する分類器として実装している。
# 特徴：
#   - REINFORCE：サンプル軌跡に対し、 log π(a_t|s_t) * G_t を損失として最小化する（符号は負）。
#   - 報酬の標準化：バッチ内の報酬を正規化して分散を抑制し、学習安定化を図る。
#   - StandardScaler による状態の前処理：スケール差による学習不安定を抑える。
#   - TensorFlow v1 互換のグラフ実行：placeholder/optimizer.get_updates/K.backend.function を用いる。
# 注意：
#   - sklearn.externals.joblib は新しめのscikit-learnでは非推奨である（joblibを直接importするのが一般的）。
#   - TensorFlow 2.x 環境では eager を無効化して v1 風に使っている（tf.compat.v1.disable_eager_execution）。
# ===============================================

import os
import random
import argparse
import numpy as np
from sklearn.preprocessing import (
    StandardScaler,
)  # 特徴量の標準化（平均0, 分散1）用スケーラ
from sklearn.externals import joblib  # (注) 近年は from joblib import dump, load を推奨
import tensorflow as tf
from tensorflow.python import keras as K  # Keras（TFバンドル版）を内部API経由で利用
import gym
from fn_framework import (
    FNAgent,
    Trainer,
    Observer,
    Experience,
)  # 本リポジトリの共通基盤（Agent/Trainer/Observerなど）

tf.compat.v1.disable_eager_execution()  # TF2系でもv1グラフ実行モードを強制する


# ---------------------------------------------------------------
# 方策勾配エージェント
#   - ε-greedy のような価値ベース探索は使わず、方策が確率を出力する（self policy）
#   - 学習時には REINFORCE の目的関数： E[ -log π(a|s) * G ] を最小化
# ---------------------------------------------------------------
class PolicyGradientAgent(FNAgent):

    def __init__(self, actions):
        # PolicyGradientAgent uses self policy (doesn't use epsilon).
        super().__init__(
            epsilon=0.0, actions=actions
        )  # εは使わないため 0 固定。actions は行動IDのリスト
        self.estimate_probs = True  # FNAgent 規約：推定が確率であることを示すフラグ
        self.scaler = StandardScaler()  # 観測状態の標準化器（学習前にfitする）
        self._updater = None  # Keras backend の更新用関数ハンドルを後で格納

    def save(self, model_path):
        # モデル（NNパラメータ）保存に加え、前処理（StandardScaler）も併せて保存して再現性を担保する
        super().save(model_path)
        joblib.dump(
            self.scaler, self.scaler_path(model_path)
        )  # スケーラを別ファイルにシリアライズ

    @classmethod
    def load(cls, env, model_path):
        # 保存済みモデルとスケーラを読み出して推論可能なエージェントを再構築する
        actions = list(range(env.action_space.n))  # 行動空間サイズを環境から決定
        agent = cls(actions)
        agent.model = K.models.load_model(model_path)  # Keras モデルのロード
        agent.initialized = True
        agent.scaler = joblib.load(
            agent.scaler_path(model_path)
        )  # 学習時のスケーラを復元（入力正規化の一貫性）
        return agent

    def scaler_path(self, model_path):
        # モデルパスから拡張子を差し替えてスケーラ保存用のファイル名を生成する
        fname, _ = os.path.splitext(model_path)
        fname += "_scaler.pkl"
        return fname

    def initialize(self, experiences, optimizer):
        # 収集済みの経験（状態）から入力次元を推定し、方策NNを構築・初期化する
        states = np.vstack([e.s for e in experiences])  # shape: (N, state_dim)
        feature_size = states.shape[1]  # 入力次元を自動算出
        # NNアーキテクチャ：2層MLP（ReLU）→ softmax（各行動の確率）
        self.model = K.models.Sequential(
            [
                K.layers.Dense(
                    10, activation="relu", input_shape=(feature_size,)
                ),  # 隠れ層1
                K.layers.Dense(10, activation="relu"),  # 隠れ層2
                K.layers.Dense(
                    len(self.actions), activation="softmax"
                ),  # 出力：行動確率分布
            ]
        )
        self.set_updater(optimizer)  # 後述：loss構築と parameter update 用関数を用意
        self.scaler.fit(states)  # 前処理スケーラを学習（平均・分散を記録）
        self.initialized = True
        print("Done initialization. From now, begin training!")

    def set_updater(self, optimizer):
        # REINFORCE の損失を定義し、Keras backend の function として更新関数を作る
        actions = tf.compat.v1.placeholder(
            shape=(None), dtype="int32"
        )  # 行動 a_t の整数ID（バッチ）
        rewards = tf.compat.v1.placeholder(
            shape=(None), dtype="float32"
        )  # 割引和報酬 G_t （バッチ）
        one_hot_actions = tf.one_hot(
            actions, len(self.actions), axis=1
        )  # 行動を one-hot に変換
        action_probs = self.model.output  # NN出力：各行動の確率 π(a|s;θ)
        # 自分が選んだ a_t に対応する確率成分だけを抽出（∑ one_hot * probs）
        selected_action_probs = tf.reduce_sum(one_hot_actions * action_probs, axis=1)
        clipped = tf.clip_by_value(
            selected_action_probs, 1e-10, 1.0
        )  # log(0)防止のためクリップ
        loss = -tf.math.log(clipped) * rewards  # REINFORCE の負の対数尤度重み付き損失
        loss = tf.reduce_mean(loss)  # ミニバッチ平均

        # Keras Optimizer（TF v1互換）から重み更新オペレーションを取得
        updates = optimizer.get_updates(loss=loss, params=self.model.trainable_weights)
        # Keras backend.function で「入力→(損失, 更新)」の実行関数を構築
        self._updater = K.backend.function(
            inputs=[
                self.model.input,  # 正規化済み状態のバッチ
                actions,
                rewards,
            ],  # 行動ID と 割引和報酬
            outputs=[loss],  # 返り値として損失を出す
            updates=updates,
        )  # 呼び出し時にパラメータ更新が走る

    def estimate(self, s):
        # 推論：状態 s（2次元配列想定）を標準化→NNで各行動の確率を得る
        normalized = self.scaler.transform(s)  # 学習時スケールで正規化
        action_probs = self.model.predict(normalized)[0]  # shape: (n_actions,)
        return action_probs

    def update(self, states, actions, rewards):
        # 学習：バッチ（states, actions, rewards）を受け取り、1ステップの更新を実行
        normalizeds = self.scaler.transform(states)  # 状態を標準化
        actions = np.array(actions)
        rewards = np.array(rewards)
        self._updater(
            [normalizeds, actions, rewards]
        )  # backend.function を実行（重み更新）


# ---------------------------------------------------------------
# 観測ラッパ：CartPole の観測ベクトルを (1, -1) 形状に整えるだけの薄いラッパ
# ---------------------------------------------------------------
class CartPoleObserver(Observer):

    def transform(self, state):
        # 環境の観測（list/1D array）を (1, feature_dim) にreshape して NN 入力形状に合わせる
        return np.array(state).reshape((1, -1))


# ---------------------------------------------------------------
# 学習の司令塔：バッファ・バッチ作成・エピソード境界での処理・ロギングなど
# ---------------------------------------------------------------
class PolicyGradientTrainer(Trainer):

    def __init__(
        self, buffer_size=256, batch_size=32, gamma=0.9, report_interval=10, log_dir=""
    ):
        # buffer_size: 初期化に必要な最小経験数（初期NN構築のため）
        # batch_size : 学習1回あたりに使うサンプル数
        # gamma      : 割引率（将来報酬の現在価値）
        # report_interval: 何エピソードごとに報酬統計を表示するか
        super().__init__(buffer_size, batch_size, gamma, report_interval, log_dir)

    def train(self, env, episode_count=220, initial_count=-1, render=False):
        # エージェントを生成し、共通の train_loop をまわす
        actions = list(
            range(env.action_space.n)
        )  # CartPoleは2アクション（左右）だが一般化
        agent = PolicyGradientAgent(actions)
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    def episode_begin(self, episode, agent):
        # エピソード開始時：すでにNN初期化済みなら、エピソード蓄積用の self.experiences をクリア
        if agent.initialized:
            self.experiences = []

    def make_batch(self, policy_experiences):
        # REINFORCE更新用の（状態, 行動, 割引和報酬）バッチをサンプリングする
        length = min(
            self.batch_size, len(policy_experiences)
        )  # バッチサイズとサンプル数の小さい方
        batch = random.sample(
            policy_experiences, length
        )  # ランダムサンプル（経験の順序バイアスを避ける）
        states = np.vstack([e.s for e in batch])  # shape: (B, state_dim)
        actions = [e.a for e in batch]  # 長さBの行動IDリスト
        rewards = [e.r for e in batch]  # 長さBの割引和報酬
        # 報酬の標準化（平均0, 分散1）：勾配のスケールを整え、学習安定化を図る（ベースラインの簡易代替）
        scaler = StandardScaler()
        rewards = np.array(rewards).reshape((-1, 1))
        rewards = scaler.fit_transform(rewards).flatten()
        return states, actions, rewards

    def episode_end(self, episode, step_count, agent):
        # エピソード終了時：合計報酬をロギングし、初期化 or 学習のいずれかを行う
        rewards = [e.r for e in self.get_recent(step_count)]  # 当該エピソードの逐次報酬
        self.reward_log.append(sum(rewards))  # 可視化用に合計報酬を蓄積

        if not agent.initialized:
            # まだNN未初期化：十分なサンプルが貯まったら初期化を行う
            if len(self.experiences) == self.buffer_size:
                optimizer = K.optimizers.Adam(
                    lr=0.01
                )  # 学習率0.01のAdam（v1/Keras互換）
                agent.initialize(
                    self.experiences, optimizer
                )  # NN構築＋scaler学習＋updater構築
                self.training = True
        else:
            # 既に初期化済：REINFORCEのための割引和報酬 G_t を各時刻tに対して計算
            policy_experiences = []
            for t, e in enumerate(self.experiences):
                s, a, r, n_s, d = e
                # G_t = Σ_{i=0..} γ^i * r_{t+i} を直接計算（末尾までの報酬列に指数重み）
                d_r = [_r * (self.gamma**i) for i, _r in enumerate(rewards[t:])]
                d_r = sum(d_r)
                # Experience の r を G_t へ置き換えた新しいサンプルを作る
                d_e = Experience(s, a, d_r, n_s, d)
                policy_experiences.append(d_e)

            # バッチを作って1回の更新を実行
            agent.update(*self.make_batch(policy_experiences))

        # 規定エピソード間隔で最近の報酬統計を出力（平均±分散などは logger.describe が担当）
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval :]
            self.logger.describe("reward", recent_rewards, episode=episode)


# ---------------------------------------------------------------
# エントリポイント：--play 指定時は保存済みモデルでプレイ、未指定なら学習→保存まで実行
# ---------------------------------------------------------------
def main(play):
    env = CartPoleObserver(
        gym.make("CartPole-v0")
    )  # 環境生成＋観測ラッパで入力形状を統一
    trainer = PolicyGradientTrainer()
    path = trainer.logger.path_of(
        "policy_gradient_agent.h5"
    )  # ログ保存ディレクトリ配下のモデルパスを取得

    if play:
        # 事前学習済みモデルをロードしてプレイ（評価）するモード
        agent = PolicyGradientAgent.load(env, path)
        agent.play(env)
    else:
        # 学習モード：学習→学習曲線の描画→モデルとスケーラの保存
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log, trainer.report_interval)
        trained.save(path)


# ---------------------------------------------------------------
# 直接実行時のCLI処理
#   --play : プレイ（評価）モード。指定なし：学習モード。
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG Agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")

    args = parser.parse_args()
    main(args.play)
