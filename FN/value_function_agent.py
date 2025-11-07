# ===============================================
# 値関数近似（Value Function Approximation）による CartPole 制御（説明コメント付き）
# ---------------------------------------------------------------
# ・学習対象  : OpenAI Gym CartPole-v0
# ・近似器    : scikit-learn の MLPRegressor（2層 MLP）＋ StandardScaler
# ・アルゴリズム概要:
#    - Q(s, a) を関数近似（回帰）で直接学習する（状態→各行動のQ値ベクトルを出力）
#    - 経験（s, a, r, s', done）から TD 目標 y = r + γ max_a' Q(s', a') を構成し、
#      現在の推定 Q(s, ·) のうち選択行動 a の成分を TD 目標で置換して回帰学習
# ・設計      :
#    - FNAgent   : 探索制御（ε-greedy）・推定・更新の抽象を定義
#    - Trainer   : 経験蓄積・学習ループ・ロギングの制御
#    - Observer  : 環境の観測を学習器の入力形状へ変換
# ・注意      :
#    - `from sklearn.externals import joblib` は新しめの scikit-learn では削除済みである。
#      実運用では `import joblib` を推奨（本コードは元の形を尊重しつつ後方互換のフォールバックを用意）。
# ===============================================

import random
import argparse
import numpy as np
from sklearn.neural_network import MLPRegressor  # 多層パーセプトロン（回帰器）
from sklearn.preprocessing import StandardScaler  # 特徴量のスケーリング
from sklearn.pipeline import Pipeline  # 前処理＋推定器の合成

# === 【修正点】joblib の import を新方式へ（後方互換フォールバック付き） ===
try:
    import joblib  # 推奨：scikit-learn≥0.21以降はこちらを使用
except Exception:
    # きわめて古い scikit-learn のみ互換目的でこちらを使用（現環境では通常不要）
    from sklearn.externals import joblib  # 非推奨（互換性注意）

import gym
from fn_framework import (
    FNAgent,
    Trainer,
    Observer,
)  # 既存フレームワーク（本リポジトリ内）


# ---------------------------------------------------------------
# 値関数エージェント：Q(s, a) を回帰で近似
# ---------------------------------------------------------------
class ValueFunctionAgent(FNAgent):
    # モデルの永続化（scikit-learn Pipeline をそのまま保存）
    def save(self, model_path):
        joblib.dump(self.model, model_path)

    # 永続化モデルの復元（推論専用の軽探索 ε を既定に付与）
    @classmethod
    def load(cls, env, model_path, epsilon=0.0001):
        actions = list(
            range(env.action_space.n)
        )  # 離散行動空間を 0..n-1 としてリスト化
        agent = cls(epsilon, actions)
        agent.model = joblib.load(model_path)
        agent.initialized = True
        return agent

    # 初期化：経験バッファを用いて前処理（スケーラ）を fit し、回避的に partial_fit を1回走らせる
    def initialize(self, experiences):
        scaler = StandardScaler()
        # ★ max_iter=1 としており、学習は partial_fit の反復で進める前提
        estimator = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1)
        # 入力スケーリング → MLP のパイプライン
        self.model = Pipeline([("scaler", scaler), ("estimator", estimator)])

        # 経験から状態のみを縦結合し、スケーラに適合（平均・分散を学習）
        states = np.vstack([e.s for e in experiences])
        self.model.named_steps["scaler"].fit(states)

        # 回帰器は学習前 predict 不可のため、ダミー 1 バッチで形を作る（warm-start）
        self.update([experiences[0]], gamma=0)
        self.initialized = True
        print("Done initialization. From now, begin training!")

    # 単一状態 s に対する推定 Q(s, ·) のうち、行動 a を選ぶためのスカラーを返す設計
    # （本実装では predict がベクトルを返す想定のため、[0] で先頭サンプルを取り出す）
    def estimate(self, s):
        estimated = self.model.predict(s)[0]
        return estimated

    # 複数状態の一括推定ユーティリティ
    def _predict(self, states):
        if self.initialized:
            predicteds = self.model.predict(states)  # 形状: (batch, |A|)
        else:
            # 未初期化時はランダム値で Q ベクトルを擬似生成（探索のための仮推定）
            size = len(self.actions) * len(states)
            predicteds = np.random.uniform(size=size)
            predicteds = predicteds.reshape((-1, len(self.actions)))
        return predicteds

    # TD 目標に基づく回帰更新
    # experiences: ミニバッチ（Experience のリスト）
    # γ: 割引率
    def update(self, experiences, gamma):
        # 現在状態と次状態を行列へ（predict のため 2D 形状）
        states = np.vstack([e.s for e in experiences])  # (B, state_dim)
        n_states = np.vstack([e.n_s for e in experiences])  # (B, state_dim)

        # 現在推定 Q(s, ·) と、次状態の Q(s', ·) を事前計算
        estimateds = self._predict(states)  # (B, |A|)
        future = self._predict(n_states)  # (B, |A|)

        # 各サンプルごとに、選択行動成分のみ TD 目標へ置換
        for i, e in enumerate(experiences):
            reward = e.r
            if not e.d:
                # 終了でなければ、ブートストラップ項を加える
                reward += gamma * np.max(future[i])
            # 目的ベクトルの行動成分を TD 目標で上書き（他成分は現推定を保持→multi-output回帰）
            estimateds[i][e.a] = reward

        # スケーリング後に部分学習（オンライン近似）
        estimateds = np.array(estimateds)  # (B, |A|)
        states = self.model.named_steps["scaler"].transform(states)  # 標準化
        self.model.named_steps["estimator"].partial_fit(states, estimateds)


# ---------------------------------------------------------------
# 観測変換：CartPole の観測ベクトルを (1, -1) へ整形（回帰器は 2D 入力を想定）
# ---------------------------------------------------------------
class CartPoleObserver(Observer):
    def transform(self, state):
        return np.array(state).reshape((1, -1))  # 形状: (1, state_dim)


# ---------------------------------------------------------------
# 学習ループ制御：経験蓄積 → 初期化 → ミニバッチ更新 → ログ出力
# ---------------------------------------------------------------
class ValueFunctionTrainer(Trainer):
    # 学習エントリポイント
    def train(
        self, env, episode_count=220, epsilon=0.1, initial_count=-1, render=False
    ):
        actions = list(range(env.action_space.n))
        agent = ValueFunctionAgent(epsilon, actions)
        # 親クラスの汎用ループへ委譲（経験が溜まると begin_train が呼ばれる）
        self.train_loop(env, agent, episode_count, initial_count, render)
        return agent

    # 経験バッファが埋まったタイミングで呼ばれ、スケーラの fit とウォームアップ fit を実施
    def begin_train(self, episode, agent):
        agent.initialize(self.experiences)

    # 各ステップ：訓練フェーズ中であれば、経験からランダムサンプリングして TD 学習
    def step(self, episode, step_count, agent, experience):
        if self.training:
            batch = random.sample(self.experiences, self.batch_size)
            agent.update(batch, self.gamma)

    # 各エピソード終了処理：報酬の合計をログし、一定間隔で統計を表示
    def episode_end(self, episode, step_count, agent):
        # 直近エピソードの総報酬 = そのエピソードに属する Experience の r を合算
        rewards = [e.r for e in self.get_recent(step_count)]
        self.reward_log.append(sum(rewards))

        # 指定間隔で平均・分散を標準出力（Logger.describe）
        if self.is_event(episode, self.report_interval):
            recent_rewards = self.reward_log[-self.report_interval :]
            self.logger.describe("reward", recent_rewards, episode=episode)


# ---------------------------------------------------------------
# 実行スクリプト部：学習 or プレイ
# ---------------------------------------------------------------
def main(play):
    # 観測は CartPole の状態ベクトル（位置・速度・角度・角速度）
    env = CartPoleObserver(gym.make("CartPole-v0"))
    trainer = ValueFunctionTrainer()
    path = trainer.logger.path_of("value_function_agent.pkl")  # 永続化ファイルのパス

    if play:
        # 既存モデルを読み込み、レンダリング付きでプレイ
        agent = ValueFunctionAgent.load(env, path)
        agent.play(env)
    else:
        # 学習を実施し、報酬履歴を可視化して保存
        trained = trainer.train(env)
        trainer.logger.plot("Rewards", trainer.reward_log, trainer.report_interval)
        trained.save(path)


# ---------------------------------------------------------------
# CLI 引数の処理：--play で推論実行、指定なしで学習
# ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VF Agent")
    parser.add_argument("--play", action="store_true", help="play with trained model")

    args = parser.parse_args()
    main(args.play)
