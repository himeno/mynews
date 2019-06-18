import MySQLdb
import json


class Database():
    def __init__(self):
        self.host = 'dbhost'
        self.user = 'dbuser'
        self.passwd = 'password'
        self.db = 'db'
        self.charset = 'utf8'

    def __conn(self):
        conn = MySQLdb.connect(
            host=self.host,
            user=self.user,
            passwd=self.passwd,
            db=self.db,
            charset=self.charset
        )
        cur = conn.cursor()
        return conn, cur

    def get_articles(self, label=(0, 1, 2), limit=0):
        """
        記事一覧を取得
        """
        conn, cur = self.__conn()
        if limit == 0:
            sql = "SELECT * FROM articles WHERE label IN %s ORDER BY id DESC"
            cur.execute(sql, (label,))
        else:
            sql = "SELECT * FROM articles WHERE label IN %s ORDER BY id DESC LIMIT %s"
            cur.execute(sql, (limit,))
        articles = cur.fetchall()
        conn.close()
        return articles

    def build_db(self, force=False):
        """
        初期DB作成
        """
        conn, cur = self.__conn()
        if force is True:
            cur.execute("DROP TABLE IF EXISTS articles")
        else:
            cur.execute("SHOW TABLES LIKE 'articles'")
            if cur.rowcount == 1:
                conn.close()
                return
        with open("models/fixtures.json", "r") as f:
            articles = json.loads(f.read())
        import_articles = []
        for article in articles:
            import_articles.append(
                (article['entry_id'], article['link'], article['summary'], article['label']))
        cur.execute("CREATE TABLE IF NOT EXISTS articles(id INT PRIMARY KEY AUTO_INCREMENT,entry_id TEXT,link TEXT,summary TEXT, label TINYINT) ENGINE=InnoDB DEFAULT CHARSET=utf8")
        sql = 'INSERT INTO articles(entry_id,link,summary,label) values (%s, %s, %s, %s)'
        cur.executemany(sql, import_articles)
        conn.commit()
        conn.close()

    def has_article(self, entry_id):
        """
        記事が登録済み確認
        """
        conn, cur = self.__conn()
        cur.execute('SELECT * FROM articles WHERE entry_id=%s', (entry_id,))
        if cur.fetchone() is None:
            return False
        return True

    def set_article(self, entry_id, link, summary, label):
        """
        記事を登録
        """
        conn, cur = self.__conn()
        cur.execute('INSERT INTO articles(entry_id,link,summary,label) VALUES (%s,%s,%s,%s)',
                    (entry_id, link, summary, label))
        conn.commit()
        conn.close()

    def update_label(self, id, label):
        """
        記事のラベルを更新
        """
        conn, cur = self.__conn()
        cur.execute('UPDATE articles SET label=%s WHERE id = %s', (label, id))
        conn.commit()
        conn.close()


if __name__ == "__main__":
    db = Database()
