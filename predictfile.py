# 参考URL
# http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
import os
from flask import Flask, flash, request, redirect, url_for
# 危険なファイル名などのチェック＆除去をしてくれる
from werkzeug.utils import secure_filename

# アップロードするファイルの置き場所を指定
UPLOAD_FOLDER = './uploads'
# 許可する拡張子を設定
ALLOWED_EXTENTIONS = set(['png', 'jpg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ファイルのアップロード可否判定関数
def allowed_file(filename):
    return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENTIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # データがあるかどうかの判定
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        # ユーザーがファイルを選択しなかった時、空文字が送られてくる
        if file.filename == '':
            flash('ファイルが選択されていません')
            return redirect(request.url)
        # ファイルが存在している＆適切なファイルの場合
        if file and allowed_file(file.filename):
            # サニタイズ処理(危険な文字などを削除)
            filename = secure_filename(file.filename)
            # アップロードされた画像を保存する
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # アップロード後のページに転送
            # 転送時に変数に値をセットして渡す
            return redirect(url_for('uploaded_file', filename=filename))
    # ファイルアップロードのHTMLソースを返す
    return '''
    <!doctype html>
    <html>
    <head>
    <meta charset="UTF-8">
        <title>ファイルをアップロードして判定しよう</title>
    </head>
    <body>
        <h1>ファイルをアップロードして判定しよう！</h1>
        <form method = post enctype = multipart/form-data>
            <p><input type=file name=file>
            <input type=submit value=Upload>
        </form>
    </body>
    </html>
    '''
