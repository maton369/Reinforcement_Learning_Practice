import numpy as np
import scipy.stats
from scipy.special import logsumexp
from planner import PolicyIterationPlanner
from tqdm import tqdm


class BayesianIRL:
    """
    Bayesian Inverse Reinforcement Learning（ベイズ的IRL）の最小実装クラス。

    目的:
      - 教師（エキスパート）の軌跡（状態sと行動aのペア列）を観測し、
        「各状態 s の報酬 R(s)」をベイズ的に推定する。

    手法の概略:
      - 事前分布: R(s) ~ Normal(prior_mean, prior_scale)
      - 尤度: 教師の行動が「Boltzmann/softmax方策」に従って選ばれたと仮定
          P(a|s; Q, eta) = exp(eta * Q(s,a)) / Σ_a' exp(eta * Q(s,a'))
        ここで Q は（推定報酬の下で最適化した）行動価値。
      - 近似推論/探索: 報酬ベクトル R にガウス摂動ノイズを加えた候補を複数サンプルし、
        それぞれの対数事後（= 尤度 + 事前）に基づく**ES（進化戦略）風のスコア加重平均**で
        R を更新する（スコア正規化あり）。
    """

    def __init__(self, env, eta=0.8, prior_mean=0.0, prior_scale=0.5):
        """
        env        : MDP環境（states, actions, shape, transit_func などを持つ）
        eta        : Boltzmann方策の温度逆数（大→決定的，小→ランダム）
        prior_mean : 報酬の事前平均
        prior_scale: 報酬の事前標準偏差
        """
        self.env = env
        # 推定報酬の下での最適方策/価値を得るために PolicyIteration を用いる
        self.planner = PolicyIterationPlanner(env)
        self.eta = eta
        self._mean = prior_mean
        self._scale = prior_scale
        # 報酬 R(s) に対する事前分布（独立同分布の正規分布を仮定）
        self.prior_dist = scipy.stats.norm(loc=prior_mean, scale=prior_scale)

    def estimate(
        self,
        trajectories,
        epoch=50,
        gamma=0.3,
        learning_rate=0.1,
        sigma=0.05,
        sample_size=20,
    ):
        """
        ベイズIRLのメインループ。

        trajectories: 教師データ。[(s,a), (s,a), ...] のリストを複数（エピソード分）持つ。
                      具体例: [ [(s0,a0),(s1,a1),...],  [(s0',a0'),...], ... ]
        epoch       : ES風更新の反復回数
        gamma       : プランナ（PolicyIteration）で用いる割引率
        learning_rate: ES更新のステップ幅
        sigma       : 報酬パラメータRに加えるガウス摂動ノイズのスケール
        sample_size : 1反復あたりのノイズサンプル数（集団サイズ）

        戻り値:
          2次元配列（env.shape）に成形した推定報酬マップ
        """
        num_states = len(self.env.states)

        # 初期報酬ベクトル R を事前分布に従ってサンプル（平均±スケールのランダム）
        reward = np.random.normal(size=num_states, loc=self._mean, scale=self._scale)

        def get_q(r, g):
            """
            与えた報酬ベクトル r に対して:
              1) プランナが参照する reward_func を差し替え
              2) Policy Iteration で価値/方策を最適化
              3) その価値 V から Q(s,a) を計算して返す
            備考:
              - planner.policy_to_q の実装詳細（πの係数の扱い等）はプランナ側の仕様に依存
            """
            self.planner.reward_func = lambda s: r[s]
            V = self.planner.plan(g)  # 方策評価+改善（割引率 g）
            Q = self.planner.policy_to_q(V, gamma)  # Q(s,a) を計算（内部仕様に注意）
            return Q

        # ES（Evolution Strategies）風の黒箱最適化ループ
        for i in range(epoch):
            # 各候補報酬 = 現在R + sigma*noise を sample_size 本生成
            noises = np.random.randn(sample_size, num_states)
            scores = []  # 各候補の対数事後（= 尤度 + 事前）を格納

            for n in tqdm(noises):
                _reward = reward + sigma * n
                Q = get_q(_reward, gamma)

                # 事前: 各状態のRに対して独立正規を仮定 → 対数尤度を総和
                # log p(R) = Σ_s log N(R_s | mean, scale^2)
                reward_prior = np.sum(self.prior_dist.logpdf(_r) for _r in _reward)

                # 尤度: 教師の選択行動が softmax(Q; eta) から生じると仮定
                # log p(D | R) = Σ_{(s,a)∈D} [ eta*Q(s,a) - log Σ_a' exp(eta*Q(s,a')) ]
                likelihood = self.calculate_likelihood(trajectories, Q)

                # 事後: log p(R | D) = log p(D | R) + log p(R) + 定数
                posterior = likelihood + reward_prior
                scores.append(posterior)

            # --- ESのスコアに基づく勾配推定（スコア正規化つき）---
            # 直感: 「良い（対数事後が高い）ノイズ方向」を平均してRを更新する
            rate = learning_rate / (sample_size * sigma)
            scores = np.array(scores)

            # スコア正規化（分散スケーリング）で安定化
            normalized_scores = (scores - scores.mean()) / (scores.std() + 1e-8)

            # 期待勾配 ≈ E[ normalized_score * noise ]
            noise = np.mean(noises * normalized_scores.reshape((-1, 1)), axis=0)

            # パラメータ更新
            reward = reward + rate * noise

            # 進捗ログ（平均対数事後）
            print("At iteration {} posterior={}.".format(i, scores.mean()))

        # 推定した報酬ベクトルを環境形状に整形して返す
        reward = reward.reshape(self.env.shape)
        return reward

    def calculate_likelihood(self, trajectories, Q):
        """
        教師データに対する対数尤度 log p(D | R) を計算する。
        仮定: 教師の行動 a は softmax(eta * Q(s,·)) に従う。

        数式:
          log p(D|R) = (1/|D|) Σ_{trajectory t∈D} Σ_{(s,a)∈t}
                         [ eta*Q(s,a) - log Σ_{a'} exp(eta*Q(s,a')) ]

        実装詳細:
          - 数値安定性のために、分母の和は scipy.special.logsumexp を使用。
          - ここでは軌跡ごとの合計を平均（mean）してスケールを揃えている。
        """
        mean_log_prob = 0.0

        for t in trajectories:
            t_log_prob = 0.0
            for s, a in t:
                # 分子: eta * Q(s,a)
                expert_value = self.eta * Q[s][a]
                # 分母: log Σ_a' exp(eta * Q(s,a')) を logsumexp で安定計算
                total = [self.eta * Q[s][_a] for _a in self.env.actions]
                t_log_prob += expert_value - logsumexp(total)
            mean_log_prob += t_log_prob

        # 軌跡数で平均化（学習率やスコアのスケール安定のため）
        mean_log_prob /= max(len(trajectories), 1)
        return mean_log_prob


if __name__ == "__main__":

    def test_estimate():
        """
        簡単な GridWorld での動作デモ:
          1) 真の報酬（グリッド値）で PolicyIteration を回し、教師軌跡 D を収集
          2) D を用いて Bayesian IRL で報酬マップを推定
          3) 推定報酬を可視化
        注意:
          - `planner.PolicyIterationPlanner` 側の Q 計算や done 判定の仕様に依存するため、
            理論的なMaxEnt/ベイズIRLと完全に一致しない振る舞いになる可能性があります。
        """
        from environment import GridWorldEnv

        env = GridWorldEnv(
            grid=[
                [0, 0, 0, 1],  # 右上がゴール(+1)
                [0, 0, 0, 0],
                [0, -1, 0, 0],  # 途中にトラップ(-1)
                [0, 0, 0, 0],
            ]
        )

        # 1) 教師方策の作成（真の報酬で最適化）
        teacher = PolicyIterationPlanner(env)
        teacher.plan()

        # 2) 教師軌跡を収集（(s,a) の列）
        trajectories = []
        print("Gather demonstrations of teacher.")
        for i in range(20):
            s = env.reset()
            done = False
            steps = []
            while not done:
                a = teacher.act(s)  # 教師の行動を選択
                steps.append((s, a))  # (状態, 行動) を保存
                n_s, r, done, _ = env.step(a)
                s = n_s
            trajectories.append(steps)

        # 3) ベイズIRLで報酬推定
        print("Estimate reward.")
        irl = BayesianIRL(env)
        rewards = irl.estimate(trajectories)
        print(rewards)
        env.plot_on_grid(rewards)

    test_estimate()
