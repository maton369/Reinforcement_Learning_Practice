# ============================================================
# MDPのプランニング（価値反復 / 方策反復）サンプル実装
# ------------------------------------------------------------
# • 想定：環境envは以下のインタフェースを持つこと
#    - env.states           : 状態集合（0..S-1 の整数ID配列）
#    - env.actions          : 行動集合（0..A-1 の整数ID配列）
#    - env.observation_space.n : 状態数 S
#    - env.action_space.n      : 行動数 A
#    - env.reset()          : エピソード初期化
#    - env.has_done(s)      : 終端状態かどうかを返す bool
#    - env.reward_func(s)   : 状態 s の即時報酬 r(s) を返す
#    - env.transit_func(s, a) -> dict[next_state] = prob
#         （MDPの遷移確率 P(s' | s, a) を返す）
#
# • 背景理論（概要）
#   - 価値反復 (Value Iteration): Bellman最適方程式に基づき、価値Vを直接反復更新し、
#       V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [ r(s') + γ (1 - done(s')) V_k(s') ]
#     の収束をもって最適価値関数を得る手法。
#
#   - 方策反復 (Policy Iteration): 現在方策πの下で方策評価（V^πを解く）と
#     方策改善（各状態で最良行動に置換）を交互に繰り返し、最適方策へ到達する手法。
#       評価: V^{π}_{k+1}(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[ r(s') + γ (1 - done(s')) V^{π}_k(s') ]
#       改善: π'(s) = argmax_a Σ_{s'} P(s'|s,a)[ r(s') + γ (1 - done(s')) V^{π}(s') ]
#
# • 注意
#   - 本コードの遷移列挙 transitions_at では、非終端なら (p, next_state, r(next_state), done(next_state?)) を返し、
#     終端なら (1.0, None, r(state), True) を返す仕様。
#   - 下記の NOTE にある通り、終端判定に next_state を用いるのが自然（Bug候補）だが、
#     元コードの挙動を変えないためロジックは維持し、コメントで指摘のみ行う。
# ============================================================

import numpy as np


class Planner:
    """
    MDPプランナの基本クラス。
    • env      : 上記インタフェースを満たす環境
    • reward_func : 即時報酬関数 r(s)。未指定なら env.reward_func を採用
    """

    def __init__(self, env, reward_func=None):
        self.env = env
        self.reward_func = reward_func
        if self.reward_func is None:
            self.reward_func = self.env.reward_func

    def initialize(self):
        """
        プランニング開始前の初期化。
        ここではエピソードを一度リセット（環境側の内部状態を初期化）している。
        """
        self.env.reset()

    def transitions_at(self, state, action):
        """
        状態 state で行動 action をとったときの遷移（確率・次状態・即時報酬・終端）を列挙する。
        • 非終端状態: env.transit_func(s,a) から得た全ての s' について
            (p, next_state, r(next_state), done(next_state?)) をyield
        • 終端状態: (1.0, None, r(state), True) を1個だけyield
        """
        reward = self.reward_func(state)
        done = self.env.has_done(state)
        transition = []
        if not done:
            transition_probs = self.env.transit_func(state, action)
            for next_state in transition_probs:
                prob = transition_probs[next_state]
                reward = self.reward_func(next_state)
                # NOTE: 多くの定義では「終端判定」は next_state に対して行うのが自然。
                #       ここでは元コードを尊重して state を参照している（潜在的バグ候補）。
                done = self.env.has_done(state)
                transition.append((prob, next_state, reward, done))
        else:
            # 終端ならば遷移先は存在せず、報酬は現状態のものをそのまま返す。
            transition.append((1.0, None, reward, done))

        for p, n_s, r, d in transition:
            yield p, n_s, r, d

    def plan(self, gamma=0.9, threshold=0.0001):
        """
        具体的なプランニング手法（価値反復/方策反復）は派生クラスで実装する。
        """
        raise Exception("Planner have to implements plan method.")


class ValueIterationPlanner(Planner):
    """
    価値反復：
      - 全状態の価値 V を反復的に更新し、差分が閾値以下になったら収束とみなす。
      - 収束後の V は最適価値関数 V* の近似。最適方策は各状態で argmax_a により得られる。
    """

    def __init__(self, env):
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        """
        • gamma     : 割引率（将来報酬の現在価値をどれだけ重視するか）
        • threshold : 1反復での最大変化量Δの停止基準（小さいほど厳密）
        戻り値: 収束した価値ベクトル V (shape: [S])
        """
        self.initialize()
        V = np.zeros(len(self.env.states))  # 価値の初期化（全ゼロ）
        while True:
            delta = 0  # 1反復における最大更新量
            for s in self.env.states:
                expected_rewards = []  # 各行動の期待値を格納
                for a in self.env.actions:
                    reward = 0.0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            # 終端：将来価値は足さず、即時報酬のみ
                            reward = r
                            continue
                        # 非終端：r + γ V(s') を確率重み p で期待値化
                        reward += p * (r + gamma * V[n_s] * (not done))
                    expected_rewards.append(reward)
                max_reward = max(expected_rewards)
                delta = max(delta, abs(max_reward - V[s]))  # 変化量を追跡
                V[s] = max_reward  # Bellman最適方程式に沿った上書き

            # 収束判定：全状態での更新が十分小さければ終了
            if delta < threshold:
                break

        return V


class PolicyIterationPlanner(Planner):
    """
    方策反復：
      1) 方策評価：固定方策 π の下で V^π を解く（ここでは反復近似）
      2) 方策改善：各状態で最良行動に置き換え、πを改善
      を収束まで繰り返す。
    """

    def __init__(self, env):
        super().__init__(env)
        self.policy = None  # π(a|s) の行列表現（各状態での確率分布）
        self._limit_count = 1000  # 評価反復の安全上限（無限ループ防止）

    def initialize(self):
        """
        初期方策の設定：各状態で全行動一様分布（等確率）から開始。
        """
        super().initialize()
        self.policy = np.ones((self.env.observation_space.n, self.env.action_space.n))
        self.policy = self.policy / self.env.action_space.n

    def policy_to_q(self, V, gamma):
        """
        与えられた価値 V に対して、各 (s,a) の行動価値 Q(s,a) を計算する。
        Q(s,a) = Σ_{s'} P(s'|s,a)[ r(s') + γ V(s') ]
        ここでは内部で現在の π(a|s) を掛けている（元実装踏襲）。
        戻り値: Q (shape: [S, A])
        """
        Q = np.zeros((self.env.observation_space.n, self.env.action_space.n))

        for s in self.env.states:
            for a in self.env.actions:
                a_p = self.policy[s][a]  # 現行方策の a の確率（元コード互換）
                for p, n_s, r, done in self.transitions_at(s, a):
                    if done:
                        Q[s][a] += p * a_p * r
                    else:
                        Q[s][a] += p * a_p * (r + gamma * V[n_s])
        return Q

    def estimate_by_policy(self, gamma, threshold):
        """
        現在の方策 π の下での方策評価：V を収束まで反復近似する。
        V(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[ r(s') + γ V(s') ]
        戻り値: 収束した V^π
        """
        V = np.zeros(self.env.observation_space.n)

        count = 0
        while True:
            delta = 0
            for s in self.env.states:
                expected_rewards = []
                for a in self.env.actions:
                    action_prob = self.policy[s][a]
                    reward = 0.0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            # 終端：即時報酬のみ
                            reward = r
                            continue
                        reward += action_prob * p * (r + gamma * V[n_s] * (not done))
                    expected_rewards.append(reward)
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value

            # 収束 or 反復上限で停止
            if delta < threshold or count > self._limit_count:
                break
            count += 1

        return V

    def act(self, s):
        """
        現在の方策 π に従った行動（最大確率の行動）を返す。
        """
        return np.argmax(self.policy[s])

    def plan(self, gamma=0.9, threshold=0.0001, keep_policy=False):
        """
        方策反復のメイン手順。
        • keep_policy=True の場合、既存の self.policy を初期方策として継続利用。
        戻り値: 改善後に得られた（最終）価値ベクトル V
        """
        if not keep_policy:
            self.initialize()

        count = 0
        while True:
            update_stable = True
            # 1) 方策評価（V^π を近似的に解く）
            V = self.estimate_by_policy(gamma, threshold)

            for s in self.env.states:
                # 現方策が選ぶ行動（最大確率のインデックス）
                policy_action = self.act(s)

                # 2) 方策改善：全行動の期待収益を比較して最良行動を選ぶ
                action_rewards = np.zeros(len(self.env.actions))
                for a in self.env.actions:
                    reward = 0.0
                    for p, n_s, r, done in self.transitions_at(s, a):
                        if n_s is None:
                            reward = r
                            continue
                        reward += p * (r + gamma * V[n_s] * (not done))
                    action_rewards[a] = reward

                best_action = np.argmax(action_rewards)
                if policy_action != best_action:
                    update_stable = False  # 改善が起きたら、なお反復を継続

                # 改善後の方策は「貪欲化」：最良行動の確率を1、それ以外を0
                self.policy[s] = np.zeros(len(self.env.actions))
                self.policy[s][best_action] = 1.0

            # 方策が全状態で変わらなければ収束
            if update_stable or count > self._limit_count:
                break
            count += 1

        return V


if __name__ == "__main__":

    def test_plan():
        """
        動作確認用の簡易テスト。
        • environment.GridWorldEnv は、上記インタフェースを満たすものを想定。
        """
        from environment import GridWorldEnv

        env = GridWorldEnv(
            grid=[
                [0, 0, 0, 1],  # +1 の終端（ゴール）を右上に配置
                [0, 0, 0, 0],
                [0, -1, 0, 0],  # -1 の終端（トラップ）を左下に配置
                [0, 0, 0, 0],
            ]
        )
        print("Value Iteration.")
        vp = ValueIterationPlanner(env)
        v = vp.plan()
        print(v.reshape(env.shape))

        print("Policy Iteration.")
        pp = PolicyIterationPlanner(env)
        v = pp.plan()
        print(v.reshape(env.shape))

        # 参考：最終価値に対する Q の合計（各状態での行動価値の総和）を表示
        q = pp.policy_to_q(v, 0.9)
        print(np.sum(q, axis=1).reshape(env.shape))

    test_plan()
