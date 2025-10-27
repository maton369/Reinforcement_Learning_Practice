# -*- coding: utf-8 -*-
"""
グリッドワールド環境の最小実装である。
確率的遷移（行動の意図した方向に move_prob、左右に (1 - move_prob)/2、逆方向は0）
と、報酬構造（通常セル: 小さな負報酬、ゴール: +1、トラップ: -1、ブロック: 侵入不可）を備える。

・セル属性:
    0 : 通常セル（行動可能）
   -1 : ダメージセル（終端）
    1 : 報酬セル（終端）
    9 : ブロック（不可侵／行動不可）

・主なメソッド:
    - transit_func(state, action): 次状態分布 P(s'|s,a) を辞書で返す。
    - reward_func(state): 与えられた状態の即時報酬と終端判定を返す。
    - step(action): 1ステップ実行（サンプリングあり）→ (next_state, reward, done)
    - reset(): 初期配置（左下隅）にエージェントを戻す。

本コードは、方策反復・価値反復・モンテカルロ・TD学習等の基礎実験に適する設計である。
"""

from enum import Enum
import numpy as np


class State:
    """格子上の位置（row, column）を表す単純な状態クラスである。"""

    def __init__(self, row=-1, column=-1):
        # 行・列のインデックスで位置を表現する（0-index）
        self.row = row
        self.column = column

    def __repr__(self):
        # デバッグ時に読みやすい表記を返す
        return "<State: [{}, {}]>".format(self.row, self.column)

    def clone(self):
        # ミュータブル性の影響を避けるためのディープコピーである
        return State(self.row, self.column)

    def __hash__(self):
        # setやdictのキーとして扱えるようにハッシュ化する
        return hash((self.row, self.column))

    def __eq__(self, other):
        # 位置が一致すれば同一状態とみなす
        return self.row == other.row and self.column == other.column


class Action(Enum):
    """エージェントが取れる4方向の行動である。
    値設計に正負を持たせ、反対方向（opposite）を value の符号反転で求められるようにしている。
    """

    UP = 1
    DOWN = -1
    LEFT = 2
    RIGHT = -2


class Environment:
    """グリッドワールドのMDP環境である。
    - grid: 2次元配列（セル属性を格納）
    - move_prob: 意図した行動がそのまま実行される確率（スリップを表現）
    """

    def __init__(self, grid, move_prob=0.8):
        # gridは2次元配列。値はセル属性を意味する（ヘッダの表参照）
        self.grid = grid
        self.agent_state = State()

        # 通常セルにいるだけで小さな負報酬（時間ペナルティ）を与える。
        # これによりエージェントは素早くゴールへ向かう方策を学びやすくなる。
        self.default_reward = -0.04

        # move_probの確率で意図した方向に動く。
        # 残りの確率 (1 - move_prob) は左右方向に等確率でスリップする。
        # 反対方向にはスリップしない設計である。
        self.move_prob = move_prob
        self.reset()

    @property
    def row_length(self):
        # グリッドの行数である
        return len(self.grid)

    @property
    def column_length(self):
        # グリッドの列数である
        return len(self.grid[0])

    @property
    def actions(self):
        # 行動空間 A = {↑, ↓, ←, →} である
        return [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]

    @property
    def states(self):
        """全ての「到達可能」状態集合を返す。
        ブロック(9)セルは進入不可のため状態集合に含めない。
        """
        states = []
        for row in range(self.row_length):
            for column in range(self.column_length):
                # ブロックセルは状態から除外する
                if self.grid[row][column] != 9:
                    states.append(State(row, column))
        return states

    def transit_func(self, state, action):
        """遷移確率分布 P(s'|s,a) を {next_state: prob} の辞書で返す。
        - 終端状態からは行動できないため、空辞書を返す。
        - スリップにより異なる行動に確率が割り当てられる。
        - 境界やブロックにぶつかって元の場所に留まる場合、確率は同一s'に加算される。
        """
        transition_probs = {}
        if not self.can_action_at(state):
            # すでに終端セル上にいる場合は遷移なし
            return transition_probs

        # 反対方向（例: UPの反対はDOWN）をEnum値の符号反転で得る
        opposite_direction = Action(action.value * -1)

        for a in self.actions:
            prob = 0
            if a == action:
                # 意図した方向へ move_prob の確率で進む
                prob = self.move_prob
            elif a != opposite_direction:
                # 反対方向以外（左右）に等確率でスリップ
                prob = (1 - self.move_prob) / 2

            # a を実行した場合の（境界・ブロック考慮済みの）次状態を求める
            next_state = self._move(state, a)

            # 同一の次状態が複数行動で到達しうる（壁で弾かれて据え置き等）ので確率を合算する
            if next_state not in transition_probs:
                transition_probs[next_state] = prob
            else:
                transition_probs[next_state] += prob

        return transition_probs

    def can_action_at(self, state):
        """その状態で行動可能かを返す。
        通常セル(0)のみTrue。終端セル(1, -1)やブロック(9)ではFalseである。
        """
        if self.grid[state.row][state.column] == 0:
            return True
        else:
            return False

    def _move(self, state, action):
        """単一行動 a に対する決定論的な移動結果を返す（境界・ブロックで据え置き）。
        行動不可能な状態から呼ばれた場合は例外を投げる。
        """
        if not self.can_action_at(state):
            raise Exception("Can't move from here!")

        next_state = state.clone()

        # 行動を反映して座標を更新する
        if action == Action.UP:
            next_state.row -= 1
        elif action == Action.DOWN:
            next_state.row += 1
        elif action == Action.LEFT:
            next_state.column -= 1
        elif action == Action.RIGHT:
            next_state.column += 1

        # グリッド外に出る場合は元の位置に据え置く
        if not (0 <= next_state.row < self.row_length):
            next_state = state
        if not (0 <= next_state.column < self.column_length):
            next_state = state

        # ブロック(9)に当たった場合も据え置き
        if self.grid[next_state.row][next_state.column] == 9:
            next_state = state

        return next_state

    def reward_func(self, state):
        """与えられた状態の即時報酬 r(s) と終端フラグ done を返す。
        通常セル: default_reward（負）
        報酬セル(1): +1（終端）
        ダメージセル(-1): -1（終端）
        """
        reward = self.default_reward
        done = False

        # 次状態の属性で報酬と終端を判定する
        attribute = self.grid[state.row][state.column]
        if attribute == 1:
            # ゴール報酬でゲーム終了
            reward = 1
            done = True
        elif attribute == -1:
            # ダメージでゲーム終了
            reward = -1
            done = True

        return reward, done

    def reset(self):
        """初期化である。エージェントを左下隅（最下行・最左列）に配置する。"""
        self.agent_state = State(self.row_length - 1, 0)
        return self.agent_state

    def step(self, action):
        """1ステップ実行する（Gym風API）。
        1) 確率的に次状態をサンプル
        2) 報酬と終端を評価
        3) 内部状態を更新
        戻り値: (next_state, reward, done)
        """
        next_state, reward, done = self.transit(self.agent_state, action)
        if next_state is not None:
            # 終端でない限りエージェントの内部状態を更新する
            self.agent_state = next_state

        return next_state, reward, done

    def transit(self, state, action):
        """transit_func で得た分布から次状態をサンプリングし、報酬・終端を返す。
        - 終端状態 s に対して行動した場合、(None, None, True) を返す。
        """
        transition_probs = self.transit_func(state, action)
        if len(transition_probs) == 0:
            # すでに終端にいるため、これ以上の遷移はない
            return None, None, True

        # サンプリング用に状態のリストと確率のリストを作成する
        next_states = []
        probs = []
        for s in transition_probs:
            next_states.append(s)
            probs.append(transition_probs[s])

        # numpy.random.choice はオブジェクトも選択可能である（dtype=objectとして扱われる）
        next_state = np.random.choice(next_states, p=probs)

        # 次状態に基づき報酬と終端を評価する
        reward, done = self.reward_func(next_state)
        return next_state, reward, done
