# 参考URL
# http://flask.pocoo.org/docs/1.0/patterns/fileuploads/
import os
from flask import Flask, flash, request, redirect, url_for
# 危険なファイル名などのチェック＆除去をしてくれる
from werkzeug.utils import secure_filename

from keras.models import Sequential, load_model
import keras, sys
import numpy as np
from PIL import Image


classes = ["monkey","boar","crow"]
num_classes = len(classes)

# 計算時間の短縮のために画像の縮小をする
image_size = 50

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

            # アップロードした画像の保存先を求める
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # アップロードされた画像を保存する
            file.save(filepath)

            # モデルのロード
            model = load_model('./animal_cnn_aug.h5')

            # コマンドライン引数から画像名を取得し、読み込む
            image = Image.open(filepath)
            # 画像をRGB値に変換
            image = image.convert('RGB')
            # 画像をリサイズ
            image = image.resize((image_size, image_size))
            # numpyの数字列として画像を変換
            data = np.asarray(image)
            X = []
            X.append(data)
            # numpyの配列に変換
            X = np.array(X)

            # 結果を格納する変数に最初の結果を入れる
            result = model.predict([X])[0]
            # 一番値の大きい配列の添え字を取得
            predicted = result.argmax()
            # 一致する確率
            percentage = int(result[predicted] * 100)

            return classes[predicted] + ": " + str(percentage) + "%"

            # # アップロード後のページに転送
            # # 転送時に変数に値をセットして渡す
            # return redirect(url_for('uploaded_file', filename=filename))
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

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
