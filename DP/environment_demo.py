# -*- coding: utf-8 -*-
"""
ランダム方策エージェントの最小例である。
前提として、同ディレクトリの environment.py にグリッドワールド環境（Environment）が定義されている想定である。
本エージェントは環境の行動集合から一様ランダムに行動を選択し、エピソードを実行するのみである。
学習（値関数推定や方策改善）は行わないため、ベースラインとして利用できる。
"""

import random
from environment import Environment


class Agent:
    """ランダム方策エージェントである。
    - env.actions から行動集合を取得し、policy() で等確率にサンプルするだけの実装である。
    """

    def __init__(self, env):
        # 環境が提供する行動空間（UP, DOWN, LEFT, RIGHT）を保持する
        self.actions = env.actions

    def policy(self, state):
        """方策 π(a|s) を表す関数である。
        ここでは状態 s に依存せず、一様ランダムに行動 a を選ぶ（π は定数方策）である。
        """
        return random.choice(self.actions)


def main():
    # グリッドワールドの定義である。
    # セル属性の意味:
    #   0: 通常セル（行動可能・小さい負報酬）
    #   1: 報酬セル（+1 の終端）
    #  -1: ダメージセル（-1 の終端）
    #   9: ブロック（侵入不可・状態集合に含めない）
    grid = [[0, 0, 0, 1], [0, 9, 0, -1], [0, 0, 0, 0]]

    # 環境を生成する。デフォルトでは move_prob=0.8（意図通りに動ける確率）である。
    # 残り 0.2 は左右方向へのスリップとして扱われ、確率的遷移となる設計である。
    env = Environment(grid)

    # ランダム方策エージェントを生成する。
    agent = Agent(env)

    # 10エピソード試行する。
    for i in range(10):
        # 環境を初期化し、エージェントの開始位置（左下隅）を得る。
        state = env.reset()
        total_reward = 0  # エピソード累積報酬である。
        done = False  # 終端フラグである。

        # 終端（報酬セル or ダメージセル）に到達するまで1ステップずつ進める。
        while not done:
            # ランダム方策に従って行動を選択する（状態 s に依存しない）。
            action = agent.policy(state)

            # 環境を1ステップ進める。内部で確率的に次状態がサンプリングされる。
            # 戻り値: 次状態, 即時報酬, 終端フラグ
            next_state, reward, done = env.step(action)

            # 累積報酬を更新する。通常セルの小さい負報酬が蓄積するため、回り道は損である。
            total_reward += reward

            # 次ループのために状態を更新する。
            state = next_state

        # 各エピソード終了後の累積報酬を表示する（確率的遷移のため結果は乱数に依存する）。
        print("Episode {}: Agent gets {} reward.".format(i, total_reward))


if __name__ == "__main__":
    # メインルーチンを起動する。
    main()
