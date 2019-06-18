"""
TensorFlow チュートリアル
映画レビューのテキスト分類
https://www.tensorflow.org/tutorials/keras/basic_text_classification?hl=ja

映画のレビューをそのテキストを使って肯定的か否定的かに分類
映画レビューのデータセット IMDB datasetを利用
50,000件の映画レビューを訓練用とテスト用に25,000件ずつに分割
肯定および否定的なレビューはそれぞれ同数
TensorFlowの高レベルAPIのkerasを利用
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# IMDBデータセットをダウンロード（またはキャッシュからのコピー）
# 頻繁に登場する10000語を取得
imdb = keras.datasets.imdb
(train_data, train_labels), (test_data,
                             test_labels) = imdb.load_data(num_words=10000)

# IMDBデータセットは前処理済みで、サンプルのそれぞれは映画レビューの中の単語を表す整数の配列
# ラベルは0が否定的レビュー、1が肯定的なレビュー
print("Training entries: {}, labels: {}".format(
    len(train_data), len(train_labels)))

# 最初のレビューの中身
print('最初のレビューの中身')
print(train_data[0])
print(train_labels[1])

# １つ目と２つ目のレビューの単語数
# ニューラルネットワークへの入力は同じ長さでなければいけないが、まだ違う
print('１つ目と２つ目のレビューの単語数')
print(len(train_data[0]), len(train_data[1]))

# 整数を単語に戻してみる

# 単語を整数にマッピングする辞書
word_index = imdb.get_word_index()
print(word_index)
# インデックスの最初の方は予約済み
# ４つずらして辞書を作り直す
word_index = {k: (v + 3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
# キーとバリューが逆の辞書を作成
reverse_word_index = dict([(value, key)
                           for (key, value) in word_index.items()])


def decode_review(text):
    """
    レビューを整数からテキストに戻す
    辞書に無い単語は?に置き換える
    """
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# 最初のレビューをテキストで表示
print('最初のレビューをテキストで表示')
print(decode_review(train_data[0]))


# データの準備
# レビュー（整数の配列）をテンソルに変換、長さを揃える
# パディング（0埋め）で対応
# pad_sequences word_index["<PAD"] = 0で、post（後埋め）、長さ256
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# １つ目と２つ目の訓練データの長さ確認
print('１つ目と２つ目の訓練データの長さ確認')
print(len(train_data[0]), len(train_data[1]))
print('１つ目の訓練データ')
print(train_data[0])
print('２つ目の訓練データ')
print(train_data[1])


# モデルの構築
# 入力の形式は映画レビューで使われている語彙数（10,000語）
vocab_size = 10000

model = keras.models.Sequential()
# 正の整数（インデックス）を固定次元の密ベクトルに変換
# 最初の層はEmbedding（埋め込み）層です。この層は、整数にエンコードされた語彙を受け取り、それぞれの単語インデックスに対応する埋め込みベクトルを検索します。埋め込みベクトルは、モデルの訓練の中で学習されます。ベクトル化のために、出力行列には次元が１つ追加されます。その結果、次元は、(batch, sequence, embedding)となります。
model.add(keras.layers.Embedding(vocab_size, 16))

# 時系列データのためのグローバルな平均プーリング演算
# GlobalAveragePooling1D（１次元のグローバル平均プーリング）層です。この層は、それぞれのサンプルについて、シーケンスの次元方向に平均値をもとめ、固定長のベクトルを返します。この結果、モデルは最も単純な形で、可変長の入力を扱うことができるようになります。
model.add(keras.layers.GlobalAveragePooling1D())

# 全結合 活性化関数relu
# この固定長の出力ベクトルは、16個の隠れユニットを持つ全結合（Dense）層に受け渡されます。
model.add(keras.layers.Dense(16, activation=tf.nn.relu))

# 全結合 活性化関数sigmoid
# 最後の層は、1個の出力ノードに全結合されます。シグモイド（sigmoid）活性化関数を使うことで、値は確率あるいは確信度を表す0と1の間の浮動小数点数となります。
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()

# モデルのオプティマイザ（最適化）と損失関数を設定
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 検証用データを作成
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# モデルの訓練
# 512個のサンプルからなるミニバッチ
# 40エポックモデル（40回繰り返す）
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

# モデルの評価
results = model.evaluate(test_data, test_labels)
print(results)

# history_dict = history.history
# history_dict.keys()

# import matplotlib.pyplot as plt


# acc = history.history['acc']
# val_acc = history.history['val_acc']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(1, len(acc) + 1)

# # "bo" は青いドット
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # ”b" は青い実線
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()

# plt.show()

# plt.clf()   # 図のクリア
# acc_values = history_dict['acc']
# val_acc_values = history_dict['val_acc']

# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()

# plt.show()
