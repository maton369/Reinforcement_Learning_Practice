# ============================================================
# Maximum Entropy Inverse Reinforcement Learning（MaxEnt IRL）
# ------------------------------------------------------------
# 目的：
#   教師軌跡（デモ）から「報酬関数」を推定する。
#   ここでは各状態 s の特徴ベクトル φ(s) を one-hot とみなし、
#   報酬を R(s) = θ^T φ(s) という線形モデルで近似する。
#
# 理論（ざっくり）：
#   - MaxEnt IRL は「デモに整合しつつ、不要な仮定を最小化（最大エントロピー）」する
#     軌跡分布 p(τ) ∝ exp(Σ_t R(s_t)) を仮定する。
#   - パラメータ θ の最尤推定（対数尤度勾配）で得られる更新則は
#       ∇_θ L(θ) = 𝔼_demo[Σ_t φ(s_t)] - 𝔼_πθ[Σ_t φ(s_t)]
#     すなわち、教師データの特徴量期待と、推定報酬下で最適化された方策の特徴量期待の「差」。
#   - 本実装では、右項の期待を「推定報酬で計算した方策（Policy Iteration）」の
#     前向き確率伝播で近似し、その平均特徴との差で θ を勾配上昇（learning_rate）させる。
#
# 実装の要点：
#   - planner: PolicyIterationPlanner を用いて、現在の R(s)=θ^T φ(s) で方策最適化。
#   - calculate_expected_feature: 教師軌跡からの「状態出現回数」を平均化（経験的期待）。
#   - expected_features_under_policy: 初期分布を教師軌跡から推定し、遷移確率を前向きに伝播
#     （割引はかけず、長さ方向平均をとる簡易近似）して、状態周辺分布の平均を計算。
#   - θ の更新：θ ← θ + α * ( μ_demo - μ_πθ )  （線形報酬なのでそのまま差を加算）
# ============================================================

import numpy as np
from planner import PolicyIterationPlanner
from tqdm import tqdm


class MaxEntIRL:

    def __init__(self, env):
        """
        env: GridWorld 互換（states, state_to_feature, transit_func 等が必要）
        """
        self.env = env
        self.planner = PolicyIterationPlanner(
            env
        )  # 推定報酬下で最適方策を求めるために使用

    def estimate(self, trajectories, epoch=20, learning_rate=0.01, gamma=0.9):
        """
        MaxEnt IRL のメインループ。
        trajectories: 教師の状態系列（list[list[state]]）
        epoch       : θ の更新回数
        learning_rate: 勾配上昇のステップ幅
        gamma       : 方策評価・改善時（Policy Iteration）に用いる割引率

        戻り値：
          推定された報酬（env.shape に整形した 2D 配列）
        """
        # 状態特徴行列 Φ（行：状態、列：特徴）。GridWorld(one-hot)なら単位行列に相当
        state_features = np.vstack(
            [self.env.state_to_feature(s) for s in self.env.states]
        )
        # θ をランダム初期化（次元＝特徴数）
        theta = np.random.uniform(size=state_features.shape[1])

        # 教師の「経験的」特徴期待 μ_demo（状態出現頻度の平均）
        teacher_features = self.calculate_expected_feature(trajectories)

        for _ in tqdm(range(epoch)):
            # 1) 現在の θ から状態報酬ベクトル R(s)=θ^T φ(s) を作る
            rewards = state_features.dot(theta.T)  # shape: (num_states,)

            # 2) 推定報酬下で方策を最適化（Policy Iteration）
            #    planner は env の reward_func を参照するため、差し替える
            self.planner.reward_func = lambda s: rewards[s]
            self.planner.plan(gamma=gamma)  # 方策改善→ self.planner.policy に反映

            # 3) その方策の下での「特徴期待」 μ_π を近似計算
            #    戻り値は状態辺りの周辺分布（平均）。これを特徴空間に写像
            features = self.expected_features_under_policy(
                self.planner.policy, trajectories
            )
            # μ_π を特徴空間へ（one-hot なら state_features の線形結合で OK）
            mu_pi = features.dot(state_features)  # shape: (num_features,)

            # 4) θ を勾配上昇：θ ← θ + α ( μ_demo - μ_π )
            update = teacher_features - mu_pi
            theta += learning_rate * update

        # 学習後の報酬マップを返す（描画などしやすいように 2D へ成形）
        estimated = state_features.dot(theta.T)
        estimated = estimated.reshape(self.env.shape)
        return estimated

    def calculate_expected_feature(self, trajectories):
        """
        教師データの「経験的」特徴期待（ここでは状態出現頻度の平均）を計算。
        GridWorld + one-hot 特徴では、単に各状態の出現回数を平滑化したものになる。

        戻り値:
          shape: (num_features,) ＝ (num_states,) を想定
        """
        features = np.zeros(self.env.observation_space.n)
        for t in trajectories:
            for s in t:
                features[s] += 1

        features /= len(trajectories)  # 軌跡本数で平均
        return features

    def expected_features_under_policy(self, policy, trajectories):
        """
        推定報酬下で得た方策の下における「状態周辺分布」の近似。

        手順（簡易近似）：
          1) 初期状態分布 p(s_0) を教師軌跡の先頭状態から推定
          2) 時刻 t=1..T-1 まで、p(s_t) = Σ_{s_{t-1}} p(s_{t-1}) P(s_t|s_{t-1}, a(s_{t-1}))
             を前向きに伝播
             - a(s) は最適方策の決定的行動（planner.act）で近似
          3) 各 t の p(s_t) を平均（time-average）して最終的な状態周辺分布を返す
             （真の MaxEnt では割引付きの期待や、ソフトな遷移等を考慮するが、
               ここでは簡潔な近似）

        注意：
          - 引数 policy はインタフェース上受け取るが、実装では self.planner.act を用いて
            「決定的に」行動を選択している（方策が stochastic な場合は要改善）。
          - さらに厳密には、MaxEnt IRL の期待は soft policy / soft value で求めることが多い。
            本コードは「最適決定方策 + 前向き伝播」という近似である。
        """
        t_size = len(trajectories)  # ここでは「長さ T ≈ 軌跡本数」という近似
        states = self.env.states
        transition_probs = np.zeros((t_size, len(states)))  # 各時刻の周辺分布 p_t(s)

        # 1) 初期状態分布を教師軌跡から推定
        initial_state_probs = np.zeros(len(states))
        for t in trajectories:
            initial_state_probs[t[0]] += 1
        initial_state_probs /= len(trajectories)
        transition_probs[0] = initial_state_probs

        # 2) 方策 a(s)=argmax_π の下で前向きに分布伝播
        for t in range(1, t_size):
            for prev_s in states:
                prev_prob = transition_probs[t - 1][prev_s]
                if prev_prob == 0:
                    continue
                # NOTE: policy 引数は使わず、planner.act を使って決定的に行動を選択
                a = self.planner.act(prev_s)
                probs = self.env.transit_func(prev_s, a)  # P(s'|s,a)
                for s in probs:
                    transition_probs[t][s] += prev_prob * probs[s]

        # 3) 時間平均で滑らかに（1/T * Σ_t p_t(s)）
        total = np.mean(transition_probs, axis=0)
        return total


if __name__ == "__main__":

    def test_estimate():
        """
        簡単な GridWorld で MaxEnt IRL を試すデモ。
          1) 真の報酬（グリッドの値）で PolicyIteration を回し、教師軌跡を収集
          2) その軌跡だけを見て報酬を推定（学習）
          3) 推定報酬を表示・可視化
        """
        from environment import GridWorldEnv

        env = GridWorldEnv(
            grid=[
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 0, 0],
            ]
        )

        # 1) 教師方策の作成（真の報酬で最適化）
        teacher = PolicyIterationPlanner(env)
        teacher.plan()

        # 2) 教師軌跡を収集
        trajectories = []
        print("Gather demonstrations of teacher.")
        for _ in range(20):
            s = env.reset()
            done = False
            steps = [s]
            while not done:
                a = teacher.act(s)
                n_s, r, done, _ = env.step(a)
                steps.append(n_s)
                s = n_s
            trajectories.append(steps)

        # 3) 報酬を推定（MaxEnt IRL）
        print("Estimate reward.")
        irl = MaxEntIRL(env)
        rewards = irl.estimate(trajectories, epoch=100)
        print(rewards)
        env.plot_on_grid(rewards)

    test_estimate()
