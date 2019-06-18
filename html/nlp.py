import MeCab
import pickle
import collections
import tensorflow as tf
from tensorflow import keras
import numpy as np


class Nlp():
    def __init__(self):
        self.path_model = './models/model.pickle'
        self.path_dict = './models/dictionary.pickle'
        self.path_articles = './models/articles.pickle'
        self.path_labels = './models/labels.pickle'
        self.max_len = 64
        self.vocab_size = 5000
        self.epochs = 40
        self.batch_size = 16

    def predict(self, articles):
        """
        予測
        """
        dictionary, r_dictionary = self.adjust_dict(self.get_dictionary())
        model = keras.models.load_model(self.path_model)

        # エンコード
        en_articles = [self.encode_article(r_dictionary, a) for a in articles]
        for article in articles:
            predict_data = keras.preprocessing.sequence.pad_sequences(
                en_articles,
                value=r_dictionary["<PAD>"],
                padding='post',
                maxlen=self.max_len)

        classes = model.predict_classes(predict_data)
        return classes

    def fit(self, dictionary, articles, labels):
        """
        学習
        """

        dictionary, r_dictionary = self.adjust_dict(dictionary)

        # 記事のエンコード
        articles = [self.encode_article(
            r_dictionary, a['summary']) for a in articles]

        train_data = keras.preprocessing.sequence.pad_sequences(
            articles,
            value=r_dictionary["<PAD>"],
            padding='post',
            maxlen=self.max_len)

        # モデルの構築
        # 入力の形式は映画レビューで使われている語彙数（10,000語）
        vocab_size = self.vocab_size
        model = keras.models.Sequential()
        model.add(keras.layers.Embedding(vocab_size, 16))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(16, activation=tf.nn.relu))
        model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
        model.summary()

        # モデルのオプティマイザ（最適化）と損失関数を設定
        model.compile(optimizer=tf.keras.optimizers.Adam(),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # 検証用データを作成
        x_val = train_data[:10]
        partial_x_train = train_data[10:]

        y_val = labels[:10]
        partial_y_train = labels[10:]

        # モデルの訓練
        history = model.fit(partial_x_train,
                            partial_y_train,
                            epochs=self.epochs,
                            batch_size=self.batch_size,
                            validation_data=(x_val, y_val),
                            verbose=1)

        # モデルの保存
        model.save(self.path_model)

    def adjust_dict(self, dictionary):
        """
        学習用に辞書を調整
        """
        dictionary = dict([(key, value)
                           for key, value in enumerate(dictionary)])
        reverse_dictionary = dict([(value, key)
                                   for (key, value) in dictionary.items()])
        reverse_dictionary = {k: (v + 3)
                              for k, v in reverse_dictionary.items()}
        reverse_dictionary["<PAD>"] = 0
        reverse_dictionary["<START>"] = 1
        reverse_dictionary["<UNK>"] = 2  # unknown
        reverse_dictionary["<UNUSED>"] = 3
        dictionary = dict([(key, value)
                           for key, value in enumerate(reverse_dictionary)])
        return dictionary, reverse_dictionary

    def encode_article(self, r_dictionary, article):
        """
        記事をエンコード
        """
        o_article = self.__ma(self, article)
        return [r_dictionary.get(w) if r_dictionary.get(w) is not None else 2 for w in o_article]

    def decode_article(self, dictionary, article):
        """
        記事をデコード
        """
        return [dictionary.get(k) for k in article]

    def decode_articles(text):
        """
        レビューを整数からテキストに戻す
        辞書に無い単語は?に置き換える
        """
        return ' '.join([dictionary.get(i, '?') for i in text])

    def get_dictionary(self):
        """
        辞書データを渡す
        """
        with open(self.path_dict, 'rb') as f:
            dictionary = pickle.load(f)
        return dictionary

    def get_articles(self):
        """
        記事データを渡す
        """
        with open(self.path_articles, 'rb') as f:
            articles = pickle.load(f)
        return articles

    def get_labels(self):
        """
        ラベルデータを渡す
        """
        with open(self.path_labels, 'rb') as f:
            labels = pickle.load(f)
        return labels

    def build_dictionary(self, articles):
        """
        辞書データ作成
        """
        bag_of_words = []
        clc = collections
        dic = {}
        words = []
        for words in [self.__ma(self, article['summary']) for article in articles]:
            for word in words:
                bag_of_words.append(word)
        for word in bag_of_words:
            dic[word] = clc.Counter(bag_of_words)[word]
        dic = collections.OrderedDict(
            sorted(dic.items(), key=lambda t: t[1], reverse=True))
        words = [word for word in dic]
        with open(self.path_dict, 'wb') as f:
            pickle.dump(words, f)

    def build_articles(self, dictionary, org_articles):
        """
        記事データ作成
        """
        articles = []
        for o_a in [o_a['summary'] for o_a in org_articles]:
            articles.append([dictionary.index(word)
                             for word in self.__ma(self, o_a)])
        with open(self.path_articles, 'wb') as f:
            pickle.dump(articles, f)

    def build_labels(self, articles):
        """
        ラベルデータ作成
        """
        labels = [article['label'] for article in articles]
        with open(self.path_labels, 'wb') as f:
            pickle.dump(labels, f)

    @staticmethod
    def __ma(self, doc):
        """
        文章の形態素解析
        """
        words_docs = []
        tagger = MeCab.Tagger()
        tagger.parse('')
        node = tagger.parseToNode(doc)
        while node:
            # 必要な品詞のみ抽出
            # if node.feature.split(",")[0] in ['名詞']:
            word = node.surface
            words_docs.append(word)
            node = node.next
        return words_docs


if __name__ == '__main__':
    nlp = Nlp()
    articles = [
        'Amazon（アマゾン）で毎日開催されているタイムセール。本日2019年6月13日は1,000円台のランニングアームバンドや吸水・速乾マイクロファイバータオル10枚セットなど今すぐ欲しい人気のアイテムがお得に多数登場しています。',
        'スマホを使っていなくても、電源をオフにする必要はありません。バッテリーへの影響もなく、スマホの寿命に悪影響を与える心配はありません。',
        'パソコンが重いと生産性が落ちるだけでなく、イライラしてくるもの。基本チューニングをマスターして常にサクサク動くようにしておきましょう。',
        'KGBは世界的にも有名な旧ソ連の諜報機関ですが、そこで行われていた諜報員の記憶術は現代の私たちにも参考になりそうです。'
    ]
    classes = nlp.predict(articles)
    print(classes)

    # from database import Database
    # db = Database()
    # nlp = Nlp()
    # articles = db.get_articles()
    # articles = [{'id': a[0], 'entry_id':a[1], 'link':a[2],
    #              'summary':a[3], 'label':a[4]}for a in articles]
    # nlp.build_dictionary(articles)
    # dictionary = nlp.get_dictionary()
    # nlp.build_articles(dictionary, articles)
    # nlp.build_labels(articles)
    # labels = nlp.get_labels()
    # nlp.fit(dictionary, articles, labels)
    # nlp.encode_article(dictionary, articles[0]['summary'])
