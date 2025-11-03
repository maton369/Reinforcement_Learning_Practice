class Planner:
    """
    動的計画法ベースのプランナの基底クラスである。
    すべてのPlanner系アルゴリズム（例: 価値反復・方策反復）はこのクラスを継承する。
    """

    def __init__(self, env):
        # 環境（env）オブジェクトを受け取り、保持する。
        # env は MDP 環境を表し、状態・行動・遷移関数などを提供する前提。
        self.env = env
        # 計算過程を可視化・記録するためのログを初期化する。
        self.log = []

    def initialize(self):
        # 環境を初期状態にリセットし、ログを空にする。
        self.env.reset()
        self.log = []

    def plan(self, gamma=0.9, threshold=0.0001):
        # 具象クラスで実装されるべき抽象メソッド。
        # gamma: 割引率、threshold: 収束判定の閾値。
        raise Exception("Planner have to implements plan method.")

    def transitions_at(self, state, action):
        """
        現在の状態stateで行動actionを取ったとき、
        次にどの状態にどんな確率で遷移し、どの報酬を得るかを返す。
        """
        transition_probs = self.env.transit_func(state, action)
        for next_state in transition_probs:
            prob = transition_probs[next_state]  # 遷移確率P(s'|s,a)
            reward, _ = self.env.reward_func(next_state)  # 次状態での報酬R(s')
            yield prob, next_state, reward  # 3要素(確率, 次状態, 報酬)を返す

    def dict_to_grid(self, state_reward_dict):
        """
        各状態の値を2次元グリッド形式に変換する。
        可視化・描画などの目的で使用。
        """
        grid = []
        for i in range(self.env.row_length):
            row = [0] * self.env.column_length
            grid.append(row)
        for s in state_reward_dict:
            grid[s.row][s.column] = state_reward_dict[s]
        return grid


class ValueIterationPlanner(Planner):
    """
    ベルマン最適方程式に基づく価値反復法(Value Iteration)を実装するクラス。
    方策を明示的に持たず、V(s)を更新して収束させる。
    """

    def __init__(self, env):
        # 基底クラスのコンストラクタを呼び出す。
        super().__init__(env)

    def plan(self, gamma=0.9, threshold=0.0001):
        # 価値反復法のメインループ。
        # gamma: 割引率、threshold: 収束判定用閾値。
        self.initialize()
        actions = self.env.actions  # 可能な行動集合を取得

        # 各状態の初期価値を0で初期化する。
        V = {}
        for s in self.env.states:
            V[s] = 0  # 初期値V(s)=0

        # 収束まで反復処理を行う。
        while True:
            delta = 0  # 1ステップでの最大変化量を初期化
            self.log.append(self.dict_to_grid(V))  # 現在のVをログに保存

            for s in V:
                # 終端状態や行動できない状態はスキップ。
                if not self.env.can_action_at(s):
                    continue

                expected_rewards = []  # 各行動aに対する期待値を格納
                for a in actions:
                    r = 0
                    # 各遷移先状態s'に対して、ベルマン更新式を適用
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)

                # 行動の中で最大の期待値を選択 (max_a)
                max_reward = max(expected_rewards)
                # 最大変化量deltaを更新
                delta = max(delta, abs(max_reward - V[s]))
                # 状態価値を更新
                V[s] = max_reward

            # 変化量が閾値以下なら収束とみなして終了
            if delta < threshold:
                break

        # 結果を2次元グリッドに変換して返す。
        V_grid = self.dict_to_grid(V)
        return V_grid


class PolicyIterationPlanner(Planner):
    """
    方策反復法(Policy Iteration)を実装するクラス。
    「方策評価」と「方策改善」を交互に繰り返し、最適方策へ収束させる。
    """

    def __init__(self, env):
        super().__init__(env)
        self.policy = {}  # π(a|s)を格納する辞書

    def initialize(self):
        # 環境と方策を初期化する。
        super().initialize()
        self.policy = {}
        actions = self.env.actions
        states = self.env.states
        for s in states:
            self.policy[s] = {}
            for a in actions:
                # 初期方策：全行動を一様確率で選択（探索に偏りがない）
                self.policy[s][a] = 1 / len(actions)

    def estimate_by_policy(self, gamma, threshold):
        """
        現在の方策πに基づいて、方策評価を行う。
        反復法によりV^π(s)を収束させる。
        """
        V = {}
        for s in self.env.states:
            V[s] = 0  # 初期化 V(s)=0

        while True:
            delta = 0
            for s in V:
                expected_rewards = []
                # 各行動について期待報酬を計算
                for a in self.policy[s]:
                    action_prob = self.policy[s][a]  # π(a|s)
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        # 方策πに基づく期待値
                        r += action_prob * prob * (reward + gamma * V[next_state])
                    expected_rewards.append(r)
                # 新しい価値V(s)を更新
                value = sum(expected_rewards)
                delta = max(delta, abs(value - V[s]))
                V[s] = value

            # 変化量が閾値以下なら収束
            if delta < threshold:
                break
        return V

    def plan(self, gamma=0.9, threshold=0.0001):
        # 方策反復法のメインループ。
        self.initialize()
        states = self.env.states
        actions = self.env.actions

        # 与えられた行動価値辞書から最大値の行動を返すユーティリティ関数
        def take_max_action(action_value_dict):
            return max(action_value_dict, key=action_value_dict.get)

        while True:
            update_stable = True  # 方策が安定（更新されない）かどうか

            # 現在の方策πに基づいてV^πを推定
            V = self.estimate_by_policy(gamma, threshold)
            self.log.append(self.dict_to_grid(V))  # ログに記録

            for s in states:
                # 現方策で最も確率の高い行動（代表行動）を取得
                policy_action = take_max_action(self.policy[s])

                # 各行動aに対する期待報酬を計算し、最良の行動を選択
                action_rewards = {}
                for a in actions:
                    r = 0
                    for prob, next_state, reward in self.transitions_at(s, a):
                        r += prob * (reward + gamma * V[next_state])
                    action_rewards[a] = r

                # 最適行動を取得
                best_action = take_max_action(action_rewards)

                # 現方策と異なれば更新が必要
                if policy_action != best_action:
                    update_stable = False

                # 方策更新：greedy（最良行動の確率1、他0）
                for a in self.policy[s]:
                    prob = 1 if a == best_action else 0
                    self.policy[s][a] = prob

            # 方策が変化しなければ（安定した場合）終了
            if update_stable:
                break

        # 最終的な状態価値をグリッド形式で返す。
        V_grid = self.dict_to_grid(V)
        return V_grid
