from bs4 import BeautifulSoup
from datetime import datetime
import urllib2
import re
import MySQLdb
import mysql.connector
import mechanize

connector = MySQLdb.connect(host="localhost", db="GCINews", user="root",passwd="", charset="utf8")
cursor = connector.cursor()
timestamp = "Wed Jun 3, 2015"
sql = "SELECT * from News where timestamp like '"
sql += "%"
sql += str(timestamp)
sql += "%"
sql += "';"
cursor.execute(sql)
News = cursor.fetchall()
#print News
for news in News:
	print news[4]
print "Please Title, if you want to read"
title = raw_input().decode("utf-8")
sql = 'SELECT bodyText from News where title like "'
sql += "%"
sql += title
sql += "%"
sql += '";'
try:
	cursor.execute(sql)
	Text = cursor.fetchall()
except:
	print sql
print len(Text[0][0][0]) 
print len(Text[0][0]) 
if len(Text[0][0][0]) >1:
	print Text[0][0][0]
else:
	print Text[0][0]


