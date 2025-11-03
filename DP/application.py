# =========================================
# Tornado Web アプリケーション層（詳細コメント＋理論解説付き）
# -----------------------------------------
# 本モジュールは、強化学習の「動的計画法（DP）」に基づく
# 価値反復（Value Iteration）/ 方策反復（Policy Iteration）を
# Web API 経由で実行し、結果（各反復の価値関数ログ）を JSON で返す。
#
# 【理論背景（有限MDPを想定）】
#   - MDP の構成要素： (S:状態, A:行動, P(s'|s,a):遷移確率, R(s):報酬, γ:割引率)
#   - 価値反復：ベルマン最適方程式を直接反復
#       V_{k+1}(s) = max_a Σ_{s'} P(s'|s,a) [ R(s') + γ V_k(s') ]
#     → 各状態で「最良行動を仮定した将来価値の期待値」を更新していく。
#   - 方策反復：方策評価と方策改善を交互に適用
#       （評価）V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [ R(s') + γ V^π(s') ]
#       （改善）π_{new}(s) = argmax_a Σ_{s'} P(s'|s,a) [ R(s') + γ V^π(s') ]
#     → 現在の方策のもとでの価値を求め（評価）、その価値に対して貪欲に方策を改善する。
#
# Web 側では、クライアントから受け取るグリッド（grid）と遷移成功確率（move_prob）を
# Environment に渡し、選択された Planner（Value/Policy）で plan() を実行、
# 反復ごとの価値関数スナップショット（grid）を log に積み上げて返却する。
# =========================================

import os
import tornado.web
import tornado.escape
from environment import (
    Environment,
)  # MDP環境：gridとmove_probから遷移P,報酬R,終端などを提供する想定
from planner import (
    ValueIterationPlanner,
    PolicyIterationPlanner,
)  # DPアルゴリズムの実装


class IndexHandler(tornado.web.RequestHandler):
    """
    ルートパスへのGETリクエストを処理するハンドラである。
    - 単純にテンプレート "index.html" を描画する。
    - UIはここから planner API (/plan) を叩くことを想定している。
    """

    def get(self):
        # templates/index.html を描画。設定は Application の settings.template_path を参照。
        self.render("index.html")


class PlanningHandler(tornado.web.RequestHandler):
    """
    DP計画要求（POST /plan）を処理するAPIハンドラである。
    期待する入力（JSON）:
      {
        "grid": 2次元配列（状態空間のレイアウト。壁/終端/通常セルなどの情報を持つ想定）,
        "plan": "value" | "policy",  # 価値反復 または 方策反復の選択
        "prob": "0.8" などの文字列（→ floatへ変換）。エージェントの意図通りに動ける確率（move_prob）
      }
    処理概要:
      1) JSONをデコードし、grid / plan_type / move_prob を抽出
      2) Environment(grid, move_prob) を構築
      3) plan_typeに応じて ValueIterationPlanner / PolicyIterationPlanner を選択
      4) planner.plan() を実行して収束させ、各反復の V を planner.log に積む
      5) 最終スナップショットも log に追加して JSON で返却
    応答（JSON）:
      { "log": [ [ [v11, v12, ...], ... ], ... ] }  # 反復ごとの2次元グリッド（価値関数）
    """

    def post(self):
        # -------------------------------
        # 1) リクエストボディ（JSON）をデコード
        #    tornado.escape.json_decode は安全なJSONデコードヘルパである。
        # -------------------------------
        data = tornado.escape.json_decode(self.request.body)
        grid = data["grid"]  # 必須：環境の状態配置を表す2次元配列
        plan_type = data["plan"]  # "value" or "policy"
        move_prob = 0.8  # 既定値：意図通りに動ける確率 P(move succeeds)

        # "prob" が与えられていれば float に変換する（無効なら既定値のまま）
        try:
            move_prob = float(data["prob"])
        except (ValueError, KeyError, TypeError):
            # 値不正/未指定の場合は既定値を使う（ここではエラーにしない設計）
            pass

        # -------------------------------
        # 2) 環境の構築：MDP の P(s'|s,a), R(s), 終端/壁の定義は Environment に委譲
        #    move_prob が高いほど行動の確実性が上がる（遷移確率が目標方向に偏る）
        # -------------------------------
        env = Environment(grid, move_prob=move_prob)

        # -------------------------------
        # 3) DPプランナの選択：価値反復 or 方策反復
        # -------------------------------
        if plan_type == "value":
            planner = ValueIterationPlanner(env)
        elif plan_type == "policy":
            planner = PolicyIterationPlanner(env)
        else:
            # 不正な plan_type の場合は 400 を返す
            self.set_status(400)
            self.write({"error": "invalid plan type", "accepted": ["value", "policy"]})
            return

        # -------------------------------
        # 4) 計画の実行：plan()
        #    - 価値反復：ベルマン最適方程式によりVを直接更新
        #    - 方策反復：V^πを評価→貪欲改善（πの更新）を繰り返す
        #    planner.log には各反復の 2D グリッド（V のスナップショット）が積まれていく実装想定
        # -------------------------------
        result = planner.plan()  # 収束後の V（2Dグリッド）が返る
        planner.log.append(
            result
        )  # 最終スナップショットを末尾に追加（クライアント可視化用）

        # -------------------------------
        # 5) 応答：全スナップショットを返す
        #    クライアント側は log[i] を順に描画することで収束過程をアニメーション表示できる。
        # -------------------------------
        self.write({"log": planner.log})


class Application(tornado.web.Application):
    """
    Tornado のアプリケーション定義である。
    - URLルーティング（handlers）
    - テンプレート/静的ファイルのパス
    - セキュリティ関連設定（cookie_secret）
    - debug モードのON/OFF
    を構成する。
    """

    def __init__(self):
        # -------------------------------
        # URL ルーティング
        #   "/"     : UI（index.html）
        #   "/plan" : DP計画API（POST）
        # -------------------------------
        handlers = [
            (r"/", IndexHandler),
            (r"/plan", PlanningHandler),
        ]

        # -------------------------------
        # アプリ設定
        #   - template_path : Jinja2風テンプレート（self.renderで使用）
        #   - static_path   : CSS/JS/画像などの静的ファイル
        #   - cookie_secret : セッション用シークレット（環境変数 SECRET_TOKEN を優先）
        #   - debug         : 自動リロード/詳細ログなど（開発時のみ True 推奨）
        # -------------------------------
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
            static_path=os.path.join(os.path.dirname(__file__), "static"),
            cookie_secret=os.environ.get(
                "SECRET_TOKEN",
                "__TODO:_GENERATE_YOUR_OWN_RANDOM_VALUE_HERE__",  # そのまま運用しないこと
            ),
            debug=True,
        )

        # 親クラスの初期化で設定を適用する。
        super(Application, self).__init__(handlers, **settings)
