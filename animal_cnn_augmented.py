from keras.models import Sequential
# 畳み込み用のライブラリをインポート
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
import keras
import numpy as np


classes = ["monkey","boar","crow"]
num_classes = len(classes)

# 計算時間の短縮のために画像の縮小をする
image_size = 50

# メインの関数を定義する
def main():
    X_train, X_test, y_train, y_test = np.load("./animal_aug.npy")
    # 正規化 0~1の範囲に収める
    X_train = X_train.astype("float") / 256
    X_test = X_test.astype("float") / 256
    # ラベルを行列表現に変換
    # (0,1,2) -> [1 0 0], [0 1 0], [0 0 1]
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)

def model_train(X, y):
    ###############
    # 層の定義を行う
    ###############

    # シーケンシャルモデルに設定
    model = Sequential()
    # `model.add()` : 畳み込みのレイヤーを追加していく
    # `Conv2D(32,(3,3)` : 32個の各フィルタを3x3
    # `padding='same'` : 畳み込み結果が同じサイズになるようにピクセルを左右に足す
    # `input_shape=X_train.shape[1:]` : 入力データ（画像）の形状
    model.add(Conv2D(32,(3,3), padding='same', input_shape=X.shape[1:]))
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

    # 呼び出す
    # batch_size: エポック（一回のトレーニング）の際に使うデータの数
    # epochs: エポックを何回行うか
    model.fit(X, y, batch_size=32, epochs=100)

    # モデルの保存
    model.save('./animal_cnn_aug.h5')

    return model

def model_eval(model, X, y):
    # テストを実行
    # verbose: 途中経過を表示する
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])

if __name__ == "__main__":
    main()
