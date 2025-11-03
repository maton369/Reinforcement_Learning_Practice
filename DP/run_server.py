import os
import tornado.ioloop
from tornado.options import define, options, parse_command_line
from application import (
    Application,
)  # Tornadoアプリケーション定義を含むモジュール（ユーザー実装想定）


# ----------------------------------------
# コマンドラインオプションの定義
# 例）python server.py --port=9000
# options.port で値を取得可能になる。
# ----------------------------------------
define("port", default=8888, help="run on the given port", type=int)


def main():
    """
    Tornado サーバーのエントリポイント。
    - コマンドライン引数を解析
    - Application を起動
    - 指定ポートでリッスン
    - IOLoop（イベントループ）を開始
    """

    # コマンドライン引数を解析し、options に値を格納する。
    # 例: `--port=9000` を指定すると options.port == 9000 となる。
    parse_command_line()

    # Tornado アプリケーションのインスタンスを生成。
    # Application クラスは routing, handler, template_path などを含む想定。
    app = Application()

    # ------------------------------------------------------------
    # ポート番号を決定
    #   - 環境変数 "PORT" が設定されていればそれを優先（Heroku等の環境向け）
    #   - 設定されていなければデフォルト 8888 を使用
    #   - もし CLI の port を優先したいなら options.port を使用可能。
    # ------------------------------------------------------------
    port = int(os.environ.get("PORT", 8888))

    # Tornado アプリケーションを指定ポートでリッスン開始。
    # 内部で HTTPServer を生成して非同期イベントループに登録する。
    app.listen(port)

    # コンソールに起動ログを出力。
    print("Run server on port: {}".format(port))

    # Tornado のメインイベントループを開始。
    # ここから非同期I/OによるHTTPサーバーが稼働し続ける。
    # 停止するには IOLoop.current().stop() を呼び出す。
    tornado.ioloop.IOLoop.current().start()


# ----------------------------------------
# メインスクリプトとして実行された場合にサーバーを起動
# ----------------------------------------
if __name__ == "__main__":
    main()
