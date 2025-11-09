# 使い方メモ:
# - このファイルは離散グリッド上の単純なMDP環境（GridWorld）を実装する。
# - OpenAI Gym の離散環境 `discrete.DiscreteEnv` を継承し、MDPの遷移確率 P, 即時報酬 r, 終端判定 done を定義する。
# - DiscreteEnv の仕様に合わせ、終端では next_state=None を使わず「自己ループ（next_state = s）」とする。

import numpy as np
from gym.envs.toy_text import discrete
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class GridWorldEnv(discrete.DiscreteEnv):

    metadata = {"render.modes": ["human", "ansi"]}

    def __init__(self, grid, move_prob=0.8, default_reward=0.0):
        # grid は 2次元配列（list/tuple/np.ndarray 可）
        # セル値の意味:
        #   0  : 通常セル（終端ではない）
        #   -1 : ダメージ（終端）
        #   1  : ゴール（終端）
        self.grid = grid
        if isinstance(grid, (list, tuple)):
            self.grid = np.array(grid)

        # 行動空間: 左/下/右/上（0/1/2/3）
        self._actions = {
            "LEFT": 0,
            "DOWN": 1,
            "RIGHT": 2,
            "UP": 3,
        }
        self.default_reward = default_reward
        self.move_prob = move_prob

        # ---------------------------------------------
        # Gym DiscreteEnv に渡すための S(状態数), A(行動数) を定義
        # ---------------------------------------------
        num_states = self.nrow * self.ncol
        num_actions = len(self._actions)

        # 初期状態分布: 左下（最下段・最左列）に確率1で配置
        initial_state_prob = np.zeros(num_states)
        initial_state_prob[self.coordinate_to_state(self.nrow - 1, 0)] = 1.0

        # ---------------------------------------------
        # 遷移確率テーブル P を構築:
        #   P[s][a] = list of [prob, next_state, reward, done]
        #   ※ DiscreteEnv 互換のため、終端でも next_state=None は使わない（自己ループ）
        # ---------------------------------------------
        P = {}
        for s in range(num_states):
            P[s] = {}
            r_s = self.reward_func(s)
            d_s = self.has_done(s)
            if d_s:
                # 終端状態: どの行動でも自己ループ
                for a in range(num_actions):
                    P[s][a] = [[1.0, s, r_s, True]]
            else:
                for a in range(num_actions):
                    P[s][a] = []
                    transition_probs = self.transit_func(s, a)
                    for n_s, p in transition_probs.items():
                        r = self.reward_func(n_s)
                        d = self.has_done(n_s)
                        P[s][a].append([p, n_s, r, d])

        self.P = P
        # DiscreteEnv のコンストラクタへ状態数・行動数・遷移表・初期分布を渡す
        super().__init__(num_states, num_actions, P, initial_state_prob)

    # -----------------------
    # グリッド幾何情報のプロパティ
    # -----------------------
    @property
    def nrow(self):
        return self.grid.shape[0]

    @property
    def ncol(self):
        return self.grid.shape[1]

    @property
    def shape(self):
        return self.grid.shape

    # Gym 互換のラッパ（リストで取得したいユースケース向け）
    @property
    def actions(self):
        return list(range(self.action_space.n))

    @property
    def states(self):
        return list(range(self.observation_space.n))

    # -----------------------
    # state <-> (row, col) 変換（行メジャー）
    # -----------------------
    def state_to_coordinate(self, s):
        # 正しい対応は divmod(s, ncol)
        row, col = divmod(s, self.ncol)
        return row, col

    def coordinate_to_state(self, row, col):
        # 正しい対応は row * ncol + col
        index = row * self.ncol + col
        return index

    # one-hot への写像（線形関数近似やNN入力用）
    def state_to_feature(self, s):
        feature = np.zeros(self.observation_space.n)
        feature[s] = 1.0
        return feature

    # -----------------------
    # MDP のコア：遷移確率 / 報酬 / 終端判定
    # -----------------------
    def transit_func(self, state, action):
        """
        指定行動 action を選んだときの「実際に適用される方向」の確率分布を生成。
        - 反対方向（opposite）は候補から除外
        - 目標方向: move_prob
        - それ以外の2方向: (1 - move_prob)/2 ずつ
        その後、グリッド境界チェックを行い、最終的な next_state の確率を集約して返す。
        """
        transition_probs = {}
        opposite_direction = (action + 2) % 4
        candidates = [a for a in range(len(self._actions)) if a != opposite_direction]

        for a in candidates:
            prob = self.move_prob if a == action else (1 - self.move_prob) / 2
            next_state = self._move(state, a)
            transition_probs[next_state] = transition_probs.get(next_state, 0.0) + prob

        return transition_probs

    def reward_func(self, state):
        """
        状態に紐づくセル値をそのまま即時報酬として返す。
        例: 1(ゴール), -1(落とし穴), 0(通常)。
        """
        row, col = self.state_to_coordinate(state)
        reward = self.grid[row][col]
        return reward

    def has_done(self, state):
        """
        終端判定: |reward| == 1 を終端とみなす（1 or -1）。
        """
        row, col = self.state_to_coordinate(state)
        reward = self.grid[row][col]
        return bool(np.abs(reward) == 1)

    # -----------------------
    # 1ステップの座標遷移（壁バンプ処理含む）
    # -----------------------
    def _move(self, state, action):
        """
        (row,col) を action に応じて1マス移動する。
        グリッド外へ出る場合はその場に留まる（壁バンプ）。
        """
        row, col = self.state_to_coordinate(state)
        next_row, next_col = row, col

        # 行動に応じて座標を更新
        if action == self._actions["LEFT"]:
            next_col -= 1
        elif action == self._actions["DOWN"]:
            next_row += 1
        elif action == self._actions["RIGHT"]:
            next_col += 1
        elif action == self._actions["UP"]:
            next_row -= 1

        # 境界チェック（はみ出しは元位置に戻す）
        if not (0 <= next_row < self.nrow):
            next_row, next_col = row, col
        if not (0 <= next_col < self.ncol):
            next_row, next_col = row, col

        next_state = self.coordinate_to_state(next_row, next_col)
        return next_state

    # -----------------------
    # 可視化ヘルパ: 値関数をグリッド上に着色表示
    # -----------------------
    def plot_on_grid(self, values):
        """
        values: shape が (nrow, ncol) または (nrow*ncol,) を想定。
        """
        if values.ndim < 2:
            values = values.reshape(self.shape)
        fig, ax = plt.subplots()
        ax.imshow(values, cmap=cm.RdYlGn)
        ax.set_xticks(np.arange(self.ncol))
        ax.set_yticks(np.arange(self.nrow))
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":

    def test_grid():
        """
        簡易自己テスト:
        - 初期位置（左下）からの壁バンプと移動挙動
        - ゴール到達時の（自己ループ, done=True, reward=1）を検証
        """
        env = GridWorldEnv(
            grid=[
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            move_prob=1.0,
        )

        s = env.reset()
        assert s == env.coordinate_to_state(env.nrow - 1, 0), "Start position mismatch"

        s2, r, d, _ = env.step(0)  # LEFT（左壁バンプ）
        assert s2 == s, "Agent should be bumped to left wall"

        s2, r, d, _ = env.step(1)  # DOWN（下壁バンプ）
        assert s2 == s, "Agent should be bumped to bottom wall"

        s, r, d, _ = env.step(2)  # RIGHT（右へ1マス）
        assert s == env.coordinate_to_state(env.nrow - 1, 1), "Agent should go to right"

        s, r, d, _ = env.step(3)  # UP（上へ1マス）
        assert s == env.coordinate_to_state(env.nrow - 2, 1), "Agent should go up"

        # ゴール（左上: (0,0)）へ到達させる
        while s != env.coordinate_to_state(0, 0):
            # 上に寄せる
            row, col = env.state_to_coordinate(s)
            if row > 0:
                s, r, d, _ = env.step(env._actions["UP"])
            elif col > 0:
                s, r, d, _ = env.step(env._actions["LEFT"])
            else:
                break

        # ゴール検証
        assert s == env.coordinate_to_state(0, 0), "Agent should be at the goal"
        assert d is True or env.has_done(s) is True, "Goal state should be terminal"
        # 終端状態でのステップは自己ループ
        s2, r2, d2, _ = env.step(env._actions["LEFT"])
        assert s2 == s and d2 is True, "Terminal state should self-loop with done=True"
        assert r2 == 1, "Agent should get reward=1 at the goal"

        print("All tests passed.")

    test_grid()
