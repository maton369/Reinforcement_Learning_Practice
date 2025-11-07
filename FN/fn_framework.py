# ===============================================
# 汎用強化学習基盤クラス群（詳細コメント付き）
# ---------------------------------------------------------------
# 本ファイルは、強化学習アルゴリズムを実装する際の基盤となるクラス群を定義している。
# - FNAgent: エージェント基底クラス（方策、推定、更新を抽象化）
# - Trainer: 学習ループ制御クラス（経験蓄積、報酬ログ、訓練制御）
# - Observer: 状態変換ラッパー（観測データの前処理抽象化）
# - Logger: TensorBoardロガー（学習経過・画像記録）
# - Experience: 経験を保存するnamedtuple（s, a, r, n_s, d）
# ===============================================

import os
import re
from collections import namedtuple
from collections import deque
import numpy as np
import tensorflow as tf
from tensorflow.python import keras as K
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 経験を表すnamedtuple
# s: 現在状態, a: 行動, r: 報酬, n_s: 次状態, d: 終了フラグ
# ---------------------------------------------------------------
Experience = namedtuple("Experience", ["s", "a", "r", "n_s", "d"])


def _reset_env(env):
    """Gym 0.21+ は (obs, info) を返すため、互換性確保で観測のみ返す。"""
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def _step_env(env, action):
    """
    Gym 0.21+ では step() が (obs, reward, terminated, truncated, info) を返す。
    旧API（obs, reward, done, info）と互換な4値に正規化する。
    """
    outcome = env.step(action)
    if len(outcome) == 5:
        obs, reward, terminated, truncated, info = outcome
        done = bool(terminated or truncated)
    else:
        obs, reward, done, info = outcome
    return obs, reward, done, info


# ===============================================================
# エージェント基底クラス（方策・推定・学習メソッドを抽象化）
# ===============================================================
class FNAgent:
    def __init__(self, epsilon, actions):
        # ε: ε-greedyの確率（ランダム行動割合）
        # actions: 行動空間のリスト
        self.epsilon = epsilon
        self.actions = actions
        self.model = None  # Kerasモデルを保持
        self.estimate_probs = False  # 方策出力が確率分布かどうか
        self.initialized = False  # モデル初期化状態フラグ

    def save(self, model_path):
        """モデルを保存する"""
        self.model.save(model_path, overwrite=True, include_optimizer=False)

    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        """保存済みモデルをロードしてFNAgentインスタンスを生成"""
        actions = list(range(env.action_space.n))
        agent = cls(epsilon, actions)
        agent.model = K.models.load_model(model_path)
        agent.initialized = True
        return agent

    def initialize(self, experiences):
        """初期化処理を抽象メソッドとして定義"""
        raise NotImplementedError("You have to implement initialize method.")

    def estimate(self, s):
        """状態 s における価値や行動分布を推定"""
        raise NotImplementedError("You have to implement estimate method.")

    def update(self, experiences, gamma):
        """エージェントの更新処理"""
        raise NotImplementedError("You have to implement update method.")

    def policy(self, s):
        """ε-greedy方策による行動選択"""
        if np.random.random() < self.epsilon or not self.initialized:
            # ランダム行動
            return np.random.randint(len(self.actions))
        else:
            # モデルによる推定値に基づく行動
            estimates = self.estimate(s)
            if self.estimate_probs:
                # 方策が確率分布の場合：確率的に行動選択
                action = np.random.choice(self.actions, size=1, p=estimates)[0]
                return action
            else:
                # 最大値行動選択
                return np.argmax(estimates)

    def play(self, env, episode_count=5, render=True):
        """現在の方策で環境を実行して報酬を観察"""
        for e in range(episode_count):
            s = _reset_env(env)
            done = False
            episode_reward = 0
            while not done:
                if render:
                    env.render()
                a = self.policy(s)
                n_state, reward, done, info = _step_env(env, a)
                episode_reward += reward
                s = n_state
            else:
                print("Get reward {}.".format(episode_reward))


# ===============================================================
# 学習ループ制御クラス（バッファ管理・ログ出力）
# ===============================================================
class Trainer:
    def __init__(
        self, buffer_size=1024, batch_size=32, gamma=0.9, report_interval=10, log_dir=""
    ):
        # 経験バッファサイズ、バッチサイズ、割引率、ログ設定など
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.report_interval = report_interval
        self.logger = Logger(log_dir, self.trainer_name)
        self.experiences = deque(maxlen=buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []

    @property
    def trainer_name(self):
        """クラス名からスネークケース名を生成（ログ用ディレクトリ名）"""
        class_name = self.__class__.__name__
        snaked = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", class_name)
        snaked = re.sub("([a-z0-9])([A-Z])", r"\1_\2", snaked).lower()
        snaked = snaked.replace("_trainer", "")
        return snaked

    def train_loop(
        self,
        env,
        agent,
        episode=200,
        initial_count=-1,
        render=False,
        observe_interval=0,
    ):
        """
        環境との相互作用を通じて学習を進めるメインループ。
        - 経験を蓄積し、指定条件で学習を開始。
        - 観察画像（フレーム）を保存するオプションあり。
        """
        self.experiences = deque(maxlen=self.buffer_size)
        self.training = False
        self.training_count = 0
        self.reward_log = []
        frames = []

        for i in range(episode):
            s = _reset_env(env)
            done = False
            step_count = 0
            self.episode_begin(i, agent)
            while not done:
                if render:
                    env.render()

                # 定期的に観察画像を保存
                if (
                    self.training
                    and observe_interval > 0
                    and (
                        self.training_count == 1
                        or self.training_count % observe_interval == 0
                    )
                ):
                    frames.append(s)

                # 行動選択と環境ステップ
                a = agent.policy(s)
                n_state, reward, done, info = _step_env(env, a)
                e = Experience(s, a, reward, n_state, done)
                self.experiences.append(e)

                # 経験が一定量溜まったら学習を開始
                if not self.training and len(self.experiences) == self.buffer_size:
                    self.begin_train(i, agent)
                    self.training = True

                self.step(i, step_count, agent, e)

                s = n_state
                step_count += 1
            else:
                self.episode_end(i, step_count, agent)

                # 一定エピソード数経過で学習を開始
                if not self.training and initial_count > 0 and i >= initial_count:
                    self.begin_train(i, agent)
                    self.training = True

                # 学習中は観察フレームをTensorBoardに出力
                if self.training:
                    if len(frames) > 0:
                        self.logger.write_image(self.training_count, frames)
                        frames = []
                    self.training_count += 1

    # 以下のメソッドは、派生クラスで各タイミングの動作を実装
    def episode_begin(self, episode, agent):
        pass

    def begin_train(self, episode, agent):
        pass

    def step(self, episode, step_count, agent, experience):
        pass

    def episode_end(self, episode, step_count, agent):
        pass

    def is_event(self, count, interval):
        """interval単位でイベントを発火させる"""
        return True if count != 0 and count % interval == 0 else False

    def get_recent(self, count):
        """直近の経験を指定数だけ取得"""
        recent = range(len(self.experiences) - count, len(self.experiences))
        return [self.experiences[i] for i in recent]


# ===============================================================
# 環境ラッパークラス（状態の変換前処理を担当）
# ===============================================================
class Observer:
    def __init__(self, env):
        self._env = env

    @property
    def action_space(self):
        """内部環境の行動空間を返す"""
        return self._env.action_space

    @property
    def observation_space(self):
        """内部環境の観測空間を返す"""
        return self._env.observation_space

    def reset(self):
        """環境リセット後の状態を変換して返す"""
        return self.transform(_reset_env(self._env))

    def render(self):
        """描画"""
        self._env.render(mode="human")

    def step(self, action):
        """1ステップ実行し、次状態・報酬・終了フラグを返す"""
        n_state, reward, done, info = _step_env(self._env, action)
        return self.transform(n_state), reward, done, info

    def transform(self, state):
        """状態変換を抽象化（画像正規化や特徴抽出など）"""
        raise NotImplementedError("You have to implement transform method.")


# ===============================================================
# TensorBoardロガー（スカラー・画像記録対応）
# ===============================================================
class Logger:
    def __init__(self, log_dir="", dir_name=""):
        """TensorBoardログディレクトリを初期化"""
        self.log_dir = log_dir
        if not log_dir:
            self.log_dir = os.path.join(os.path.dirname(__file__), "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        if dir_name:
            self.log_dir = os.path.join(self.log_dir, dir_name)
            os.makedirs(self.log_dir, exist_ok=True)

        # Eager 対応の summary writer
        self._writer = tf.summary.create_file_writer(self.log_dir)

    @property
    def writer(self):
        """TensorBoardのwriterオブジェクトを返す"""
        return self._writer

    def set_model(self, model):
        """互換用メソッド（No-op）"""
        return None

    def path_of(self, file_name):
        """ログディレクトリ内のファイルパスを生成"""
        return os.path.join(self.log_dir, file_name)

    def describe(self, name, values, episode=-1, step=-1):
        """平均値と標準偏差を出力（学習経過報告用）"""
        mean = np.round(np.mean(values), 3)
        std = np.round(np.std(values), 3)
        desc = "{} is {} (+/-{})".format(name, mean, std)
        if episode > 0:
            print("At episode {}, {}".format(episode, desc))
        elif step > 0:
            print("At step {}, {}".format(step, desc))

    def plot(self, name, values, interval=10):
        """ログ値の移動平均を可視化"""
        indices = list(range(0, len(values), interval))
        means, stds = [], []
        for i in indices:
            _values = values[i : (i + interval)]
            means.append(np.mean(_values))
            stds.append(np.std(_values))
        means = np.array(means)
        stds = np.array(stds)
        plt.figure()
        plt.title("{} History".format(name))
        plt.grid()
        plt.fill_between(indices, means - stds, means + stds, alpha=0.1, color="g")
        plt.plot(
            indices,
            means,
            "o-",
            color="g",
            label="{} per {} episode".format(name.lower(), interval),
        )
        plt.legend(loc="best")
        plt.show()

    def write(self, index, name, value):
        """スカラー値をTensorBoardに記録"""
        with self.writer.as_default():
            tf.summary.scalar(name, value, step=index)
            self.writer.flush()

    def write_image(self, index, frames):
        """フレーム画像をTensorBoardに記録（グレースケール対応）"""
        if not frames:
            return
        last_frames = [f[:, :, -1] for f in frames]
        array = np.stack(last_frames, axis=0).astype(np.float32)
        # 0-1 正規化（定数配列のときはゼロにする）
        min_val = np.min(array)
        max_val = np.max(array)
        denom = max(max_val - min_val, 1e-6)
        array = (array - min_val) / denom
        array = array[..., np.newaxis]  # (N, H, W, 1)

        with self.writer.as_default():
            tf.summary.image(
                f"frames_at_training_{index}", array, step=index, max_outputs=array.shape[0]
            )
            self.writer.flush()
