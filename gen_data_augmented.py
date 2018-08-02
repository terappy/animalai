from PIL import Image
import os, glob
import numpy as np
from sklearn import model_selection

classes = ["monkey","boar","crow"]
num_classes = len(classes)

# 計算時間の短縮のために画像の縮小をする
image_size = 50
# テストデータの数を設定
num_testdata = 50

# 画像の読み込み

X_train = []
X_test = []
Y_train = []
Y_test = []

for index, classlabel in enumerate(classes):
    photos_dir = "./" + classlabel 
    files = glob.glob(photos_dir + "/*.jpg")
    for i, file in enumerate(files):
        # 79個まで（boar画像のMAX個数）の画像を処理に使う
        if i >= 79: break
        # ファイルを開く
        image = Image.open(file)
        # RGB値に変換
        image = image.convert("RGB")
        # imageをリサイズ
        image = image.resize((image_size, image_size))
        # numpyで扱うため配列に変換
        data = np.asarray(image)

        # テストデータと学習用データを分ける
        if i < num_testdata:
            # テストデータの時
            X_test.append(data)
            Y_test.append(index)
        else:
            # 学習用データの時
            X_train.append(data)
            Y_train.append(index)

            for angle in range(-20,20,5):
                # 回転
                img_r = image.rotate(angle)
                data = np.asarray(img_r)
                X_train.append(data)
                Y_train.append(index)

                # 反転
                img_trans = image.transpose(Image.FLIP_LEFT_RIGHT)
                data = np.asarray(img_trans)
                X_train.append(data)
                Y_train.append(index)






# tensorflowが扱いやすいnumpyの配列に変換
# X = np.array(X)
# Y = np.array(Y)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(Y_train)
y_test = np.array(Y_test)

# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y)
xy = (X_train, X_test, Y_train, Y_test)
np.save("./animal_aug.npy", xy)

