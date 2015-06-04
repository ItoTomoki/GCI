#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import feedparser
from datetime import datetime
from time import mktime
 
#RSSのURL
RSS_URL  = "http://www.japantoday.com/feed/"
 
#RSSの取得
feed = feedparser.parse(RSS_URL)
 
#RSSのタイトル
print feed.feed.title
 
for entry in range(len(feed.entries)):
    #RSSの内容を一件づつ処理する
    title = feed.entries[entry].title
    link = feed.entries[entry].link
 
    #更新日を文字列として取得
    published_string = feed.entries[entry].published
 
    #更新日をdatetimeとして取得
    tmp = feed.entries[entry].published_parsed
    published_datetime = datetime.fromtimestamp(mktime(tmp))
 
    #表示
    print title
    print link
    print published_string
    print published_datetime