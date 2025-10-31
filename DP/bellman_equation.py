def V(s, gamma=0.99):
    # 状態価値関数 V(s) を定義する
    # 状態 s における報酬 R(s) と、次状態の価値の期待値を考慮して再帰的に求める。
    # gamma は割引率であり、将来の報酬の現在価値を小さくするために用いられる。
    V = R(s) + gamma * max_V_on_next_state(s)
    return V


def R(s):
    # 報酬関数 R(s) の定義
    # 状態 s が "happy_end" なら +1、"bad_end" なら -1、それ以外は 0 の報酬を返す。
    if s == "happy_end":
        return 1
    elif s == "bad_end":
        return -1
    else:
        return 0


def max_V_on_next_state(s):
    # 次状態における価値の最大値を求める関数
    # これはエージェントが「どの行動 a を選べば最も高い期待報酬を得られるか」を表す。
    # すなわち Q(s, a) = E[R + γV(s')] の最大値を取る操作に対応している。

    # 終端状態（happy_end, bad_end）の場合、これ以上報酬は得られないため 0 を返す。
    if s in ["happy_end", "bad_end"]:
        return 0

    # 取り得る行動は "up" と "down" の2種類
    actions = ["up", "down"]
    values = []

    # 各行動ごとに期待価値を計算
    for a in actions:
        transition_probs = transit_func(s, a)  # 状態遷移確率を取得
        v = 0
        # 各遷移先状態について確率加重平均をとる
        for next_state in transition_probs:
            prob = transition_probs[next_state]
            v += prob * V(next_state)  # 再帰的に V(next_state) を呼び出す
        values.append(v)
    # 最大期待価値を返す
    return max(values)


def transit_func(s, a):
    """
    遷移関数（transit function）:
    現在の状態 s と行動 a から、次に遷移する可能性のある状態とその確率を返す。

    具体的な状態遷移の仕組み:
      - 状態は文字列で表現されており、行動を '_' 区切りで付加していく。
        例:
          s='state', a='up'      → 'state_up'
          s='state_up', a='down' → 'state_up_down'
    """

    # これまでの行動履歴を "_" で分割して取り出す
    actions = s.split("_")[1:]

    # 定数定義（ゲームのルール）
    LIMIT_GAME_COUNT = 5  # 行動を取れる回数の上限（5回行動すると終了）
    HAPPY_END_BORDER = 4  # "up" の回数が 4 回以上なら成功（happy_end）
    MOVE_PROB = 0.9  # 行動が意図通りに成功する確率

    # 次状態を生成する内部関数
    def next_state(state, action):
        return "_".join([state, action])

    # もし行動回数が上限に達している場合（ゲーム終了条件）
    if len(actions) == LIMIT_GAME_COUNT:
        # "up" の回数を数えて閾値を超えていれば勝利、それ以外は敗北
        up_count = sum([1 if a == "up" else 0 for a in actions])
        state = "happy_end" if up_count >= HAPPY_END_BORDER else "bad_end"
        prob = 1.0  # 終端状態への遷移は確定的
        return {state: prob}
    else:
        # ゲームが続行中の場合：確率的に次状態へ遷移する
        opposite = "up" if a == "down" else "down"
        # MOVE_PROB の確率で希望の行動が成功
        # 1 - MOVE_PROB の確率で逆の行動が選ばれてしまう
        return {next_state(s, a): MOVE_PROB, next_state(s, opposite): 1 - MOVE_PROB}


if __name__ == "__main__":
    # テスト実行
    # 初期状態や中間状態からの状態価値を計算して出力する
    print(V("state"))  # 初期状態からの価値
    print(V("state_up_up"))  # 中間状態（上方向へ2回進んだ状態）
    print(V("state_down_down"))  # 中間状態（下方向へ2回進んだ状態）
