#!/usr/bin/env python
# coding: utf-8

import cgi
from datetime import datetime
html_body = u"""
<html>
<head>
<meta http-equiv="content-type" content="text/html;charset=utf-8" /> </head>
<body>
	<form method="POST" action="/cgi-bin/newsformJapanword2vec.py">
月を選んでください: <select name="month">
		%s
		</select>
		<input type="submit" />
	</form>
	<form method="POST" action="/cgi-bin/newsformJapanword2vec.py">
日にちを選んでください: <select name="year">
		%s
		</select>
		<input type="submit" />
	</form>
	<form method="POST" action="/cgi-bin/newsformJapanword2vec.py">
記事を選んでください: <select name="title">
		%s
		</select>
		<input type="submit" />
	</form>
	<form method="POST" action="/cgi-bin/newsformJapanword2vec.py">
ワードを選んでください: <select name="word">
		%s
		</select>
		<input type="submit" />
	</form>
%s</body>
</html>"""
moptions = ''
options=''
options2=''
content=''
moptions3 = ''
now=datetime.now()
#for m in ["06", "05"]:
for m in ["07"]:
	select=' selected="selected"'
	moptions+="<option%s>%s</option>" % (select, m)
form=cgi.FieldStorage()
month_str=form.getvalue('month', '')
if month_str == "06":
	for y in range(28,32):
		select=' selected="selected"'
		options+="<option%s>%d</option>" % (select, y)
	year_str=form.getvalue('year', '')
#elif month_str == u"Jun":
else:
	for y in range(7,14):
		select=' selected="selected"'
		options+="<option%s>%d</option>" % (select, y)
	year_str=form.getvalue('year', '')
"""
for y in range(0,4):
	select=' selected="selected"'
	options+="<option%s>%d</option>" % (select, y)
year_str=form.getvalue('year', '')
"""
#name = form[“TextArea”].value
if year_str.isdigit():
	year=int(year_str)
	content+=u"2015年"
	content+=u" "
	#content+=u"%s" % (month_str)
	content+=u"%s" % ("07")
	content+=u"月"
	content+=u" "
	content+=u"%d" % (year)
	content+=u"日"
from bs4 import BeautifulSoup
from datetime import datetime
import urllib2
import re
import MySQLdb
import mysql.connector
import mechanize

connector = MySQLdb.connect(host="localhost", db="GCINews", user="root",passwd="", charset="utf8")
cursor = connector.cursor()
#timestamp = "Wed Jun 3, 2015"

sql = "SELECT * from NewsJapan where timestamp like '"
sql += "%"
sql += content
sql += "%"
sql += "';"
content+= sql

try:
	cursor.execute(sql)
	News = cursor.fetchall()
except:
	content+=u"%s" % (sql)
	print sql
#print News
for news in News:
	select=' selected="selected"'
	options2+="<option%s>%s</option>" % (select, news[4])
	#options2+="<option%s>%s</option>" % (select, news2)
#form=cgi.FieldStorage()

title_str=form.getvalue('title', '')
title=str(title_str)

#content+=u"%s" % (title)


sql = "SELECT bodyText from NewsJapan where title like '"
sql += "%"
sql += str(title)
sql += "%"
sql += "';"
try:
	cursor.execute(sql)
	Text = cursor.fetchall()
except:
	content+=u"%s" % (sql)
content+=u"%s" % (Text[0][0])
t = (Text[0][0]).encode('utf-8')

from gensim.models import word2vec
import numpy as np
import MeCab

tagger1 = MeCab.Tagger('-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic,/usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/hatena-keyword.dic')
tagger2 = MeCab.Tagger(' -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic,/usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/hatena-keyword.dic')
result = tagger1.parse(t).split(' ')
model = word2vec.Word2Vec.load("2013.model")
"""
p = "銀行"
n = "日本"
word  = out = model.most_similar(positive=p,negative=n)
for x in out:print x[0],x[1]
"""
#voc = model.vocab.keys()
#content += u"%d" % (result.decode('utf-8'))
for m in [result]:
	select=' selected="selected"'
	moptions3+="<option%s>u'%s'</option>" % (select, m)

#form=cgi.FieldStorage()
print "Content-type: text/html;charset=utf-8¥n"
print (html_body % (moptions, options, options2, moptions3, content)).encode('utf-8')
#print (html_body % (options2, content2)).encode('utf-8')
