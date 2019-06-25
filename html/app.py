import os
from flask import *
import feedparser
import MySQLdb
import random
import json
from database import Database
from nlp import Nlp

app = Flask(__name__)
app.debug = True
db = Database()


@app.route("/", methods=["GET"])
def index(label=None):
    """
    記事一覧表示
    """
    db.build_db()
    limit = 0
    if request.args.get('label'):
        label = int(request.args.get('label'))
        labels = (label,)
    else:
        label = 'all'
        labels = (0, 1, 2)
    articles = db.get_articles(labels, limit)
    # 予測
    nlp = Nlp()
    predict_labels = [pl[0] for pl in nlp.predict([a[3] for a in articles])]
    return render_template("index.html", articles=articles, predict_labels=predict_labels, label=label)


@app.route("/edit", methods=["POST"])
def edit():
    """
    ラベルの編集
    """
    id = int(request.form['id'])
    label = int(request.form['label'])
    db.update_label(id, label)
    return redirect(request.headers.get("Referer"))


@app.route("/json_import", methods=["GET"])
def json_import():
    """
    jsonデータのインポート
    """
    with open(os.path.dirname(__file__) + "/models/fixtures.json", "r") as f:
        articles = json.loads(f.read())
    db.build_db(True)
    return redirect(request.headers.get("Referer"))


@app.route("/json_export", methods=["GET"])
def json_export():
    """
    jsonデータのエクスポート
    """
    label = (0, 1, 2)
    limit = 0
    articles = db.get_articles(label, limit)
    articles = [{'id': a[0], 'entry_id':a[1], 'link':a[2],
                 'summary': a[3], 'label': a[4]} for a in articles]
    articles_json = json.dumps(
        articles, sort_keys=True, ensure_ascii=False, indent=2)
    with open(os.path.dirname(__file__) + "/models/fixtures.json", "w") as f:
        f.write(articles_json)
    return redirect(request.headers.get("Referer"))


@app.route("/rss", methods=["POST"])
def rss():
    """
    ライフハッカー日本語版のRSSから未登録記事のみ保存
    """
    xml = 'https://www.lifehacker.jp/feed/index.xml'
    d = feedparser.parse(xml)
    entries = []
    for e in d.entries:
        if db.has_article(e.id) is False:
            db.set_article(e.id, e.link, e.summary, 2)
    return redirect(url_for('index'))


@app.route("/fit")
def fit():
    """
    学習
    """
    nlp = Nlp()
    articles = db.get_articles()
    articles = [{'id': a[0], 'entry_id':a[1], 'link':a[2],
                 'summary':a[3], 'label':a[4]}for a in articles]
    nlp.build_dictionary(articles)
    dictionary = nlp.get_dictionary()
    nlp.build_articles(dictionary, articles)
    nlp.build_labels(articles)
    labels = nlp.get_labels()
    nlp.fit(dictionary, articles, labels)

    return redirect(url_for('index'))


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    予測テスト
    """
    result = None
    if request.method == "POST":
        article = request.form['article']
        nlp = Nlp()
        classes = nlp.predict([article])
        c = classes[0][0]
        result = {'article': article, 'class': c}
    return render_template("predict.html", result=result)


@app.route("/ramdom", methods=["GET", "POST"])
def ramdom():
    """
    最新データ1割を未設定に、残り半分を肯定的と否定的に分ける
    """
    articles = db.get_articles()
    org_articles = [{'id': a[0], 'entry_id':a[1], 'link':a[2],
                     'summary': a[3], 'label': a[4]} for a in articles]
    # 1割分を分ける
    test_cnt = int(len(org_articles) / 10)
    test_articles = org_articles[0:test_cnt]
    articles = org_articles[test_cnt:len(org_articles)]
    # testデータを未設定に変更
    for article in test_articles:
        db.update_label(article['id'], 2)

    for article in articles:
        label = random.randint(0, 1)
        db.update_label(article['id'], label)

    fit()

    return redirect(url_for('index'))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
