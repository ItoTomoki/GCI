#!/usr/bin/env python
# coding: utf-8

import cgi
from datetime import datetime
html_body = u"""
<html>
<head>
<meta http-equiv="content-type" content="text/html;charset=utf-8" /> </head>
<body>
	<form method="POST" action="/cgi-bin/newsform.py">
月を選んでください: <select name="month">
		%s
		</select>
		<input type="submit" />
	</form>
	<form method="POST" action="/cgi-bin/newsform.py">
日にちを選んでください: <select name="year">
		%s
		</select>
		<input type="submit" />
	</form>
	<form method="POST" action="/cgi-bin/newsform.py">
記事を選んでください: <select name="title">
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
now=datetime.now()
for m in ["Jun", "May"]:
	select=' selected="selected"'
	moptions+="<option%s>%s</option>" % (select, m)
form=cgi.FieldStorage()
month_str=form.getvalue('month', '')
if month_str == u"May":
	for y in range(28,32):
		select=' selected="selected"'
		options+="<option%s>%d</option>" % (select, y)
	year_str=form.getvalue('year', '')
#elif month_str == u"Jun":
else:
	for y in range(0,4):
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
	content+=u"%s" % (month_str)
	content+=u" "
	content+=u"%d" % (year)
	content+=u","
	content+=u" "
	content+=u"2015"
from bs4 import BeautifulSoup
from datetime import datetime
import urllib2
import re
import MySQLdb
import mysql.connector
import mechanize
import cgi
from datetime import datetime

connector = MySQLdb.connect(host="localhost", db="GCINews", user="root",passwd="", charset="utf8")
cursor = connector.cursor()
#timestamp = "Wed Jun 3, 2015"
sql = "SELECT * from News where timestamp like '"
sql += "%"
sql += str(content)
sql += "%"
sql += "';"
cursor.execute(sql)
News = cursor.fetchall()
#print News
for news in News:
	select=' selected="selected"'
	options2+="<option%s>%s</option>" % (select, news[4])
#form=cgi.FieldStorage()
title_str=form.getvalue('title', '')
title=str(title_str)
content+=u"%s" % (title)
sql = "SELECT bodyText from News where title like '"
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
print "Content-type: text/html;charset=utf-8¥n"
print (html_body % (moptions, options, options2, content)).encode('utf-8')
#print (html_body % (options2, content2)).encode('utf-8')
