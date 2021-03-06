from keras.models import Sequential, load_model
# 畳み込み用のライブラリをインポート
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras, sys
import numpy as np
from PIL import Image

classes = ["monkey","boar","crow"]
num_classes = len(classes)

# 計算時間の短縮のために画像の縮小をする
image_size = 50

def build_model():
    ###############
    # 層の定義を行う
    ###############

    # シーケンシャルモデルに設定
    model = Sequential()
    # `model.add()` : 畳み込みのレイヤーを追加していく
    # `Conv2D(32,(3,3)` : 32個の各フィルタを3x3
    # `padding='same'` : 畳み込み結果が同じサイズになるようにピクセルを左右に足す
    # `input_shape=X_train.shape[1:]` : 入力データ（画像）の形状
    model.add(Conv2D(32,(3,3), padding='same', input_shape=(50,50,3)))
    # `Activation('relu')` : 正の値は通し，負の値は0にする
    model.add(Activation('relu'))
    model.add(Conv2D(32,(3,3)))
    model.add(Activation('relu'))
    # `MaxPooling2D(pool_size=(2,2))` : 一番大きい値を取り出す, より特徴を際立たせて取り出す
    model.add(MaxPooling2D(pool_size=(2,2)))
    # `Dropout(0.25)` : データの25%を捨て, データの偏りを減らす
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64,(3,3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64,(3,3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    # `Flatten()` : データを1列に並べるフラット処理をする
    model.add(Flatten())
    # `Dense(512)` : データの全結合の処理をする
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # `Dense(num_classes)` : 最後の出力層のノードは3つ(num_classes個分)
    model.add(Dense(num_classes))
    # `Activation('softmax')` : それぞれの画像と一致している画像を差し込むと1になる
    model.add(Activation('softmax'))

    ###

    ###############
    # 最適化の処理
    ###############

    # 最適化の手法(rmsprop)の設定
    # keras.optimizers: トレーニング時の更新アルゴリズム
    # lr: learning rate (学習率)
    # decay: lrを下げるときのレート
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # loss: 損失関数, 正解と推定値との誤差
    # optimizer: 最適化の方法
    # metrics: 評価の値，'accuracy' => どれくらい成長したか
    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])


    # モデルのロード
    model = load_model('./animal_cnn_aug.h5')

    return model

def main():
    # コマンドライン引数から画像名を取得し、読み込む
    image = Image.open(sys.argv[1])
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
    model = build_model()

    # 結果を格納する変数に最初の結果を入れる
    result = model.predict([X])[0]
    # 一番値の大きい配列の添え字を取得
    predicted = result.argmax()
    # 一致する確率
    percentage = int(result[predicted] * 100)
    # 結果を表示
    print("{0} ({1} %)".format(classes[predicted], percentage))

if __name__ == "__main__":
    main()
