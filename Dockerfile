FROM python:3.7
RUN apt-get update \
    && apt-get install -y mecab \
    && apt-get install -y libmecab-dev \
    && apt-get install -y mecab-ipadic-utf8
WORKDIR /var/www/html
ADD requirements.txt /var/www/html/
RUN pip install -r requirements.txt