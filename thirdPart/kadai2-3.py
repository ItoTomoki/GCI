#coding: utf-8
import re
import urllib2


#変数htmlには上記のHTMLがstrで代入されているとします。
#変数title, timestamp, author, author_link, bodyにそれぞれタイトル、投稿日時、著者、著者のリンク、記事本文が代入されます。

html = urllib2.urlopen("http://www.reuters.com")
print str(html)
"""
title = re.compile('\<h1\>(.+?)\<\/h1\>', re.MULTILINE|re.DOTALL).findall(html)[0]
timestamp = re.compile('\<div id="articleInfo"\>.+?\<span class="timestamp"\>(.+?)\<\/span\>', re.MULTILINE|re.DOTALL).findall(html)[0]
author_link, author = re.compile('\<div id="articleInfo"\>.+?\<span class="author"\>\<a href="(.+?)"\>(.+?)\<\/a\>\<\/span\>', re.MULTILINE|re.DOTALL).findall(html)[0]

body = re.compile('\<div id="articleText"\>(.+?)\<\/div\>', re.MULTILINE|re.DOTALL).findall(html)[0]
"""