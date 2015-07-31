#coding: utf-8
from bs4 import BeautifulSoup
from datetime import datetime
import urllib2
import re
import MySQLdb
import mysql.connector
import mechanize

connector = MySQLdb.connect(host="localhost", db="GCINews", user="root",passwd="", charset="utf8")
cursor = connector.cursor()
day = "05282015"
def SelectSportsnews(day):
	URL = "http://jp.reuters.com/news/archive/sportsNews?date=" + str(day)
	"""
	b = mechanize.Browser()
	b.open(URL)

	#変数htmlには指定のURLのHTMLが代入される。
	html = b.response().read()
	"""
	
	html = urllib2.urlopen(URL)
	soup = BeautifulSoup(html)

	#変数title, timestamp, author, author_link, bodyにそれぞれタイトル、投稿日時、著者、著者のリンク、記事本文が代入されます。
	k = 0
	j = 0
	m = day + str(j)
	for link in soup.find_all(class_= 'headlineMed standalone'):
		m = day + str(j)
		#try:
		url = "http://jp.reuters.com" + str(link.find('a').get('href'))
		Html = urllib2.urlopen(url)
		Soup = BeautifulSoup(Html)
		title = Soup.h1.find(text=True)
		timestamp = Soup.find(class_='timestampHeader').find(text=True)
			#body = Soup.meta.get('content')
		body = Soup.find(id = "resizeableText").find_all(class_ = "focusParagraph")
		bodyText = ''
		for i in body:
			bodyText = bodyText + i.find('p').find(text = True)
			print bodyText
			"""
			try:
				author  = Soup.meta.find(class_="article-info").find('a').find(text = True)
			except:
				author = ''
			try:
				author_link = Soup.find(class_="article-info").find('a').get('href')
			except:
				author_link = ''
			try:
				location = Soup.find(class_="article-info").find(class_="location").find(text = True)
			except:
				location = ''
			"""

			#title = title.replace('"','""')
			#title = title.replace("'","''")
			
		#except:
			#continue
		#sql = "INSERT INTO NewsJapan (NewsID, Timestamp, author, author_link, location, title,bodyText) VALUES("#,bodyText) VALUES("
		sql = "INSERT INTO NewsJapan (NewsID, Timestamp, title,bodyText) VALUES("
		sql += (str(m) + ",")
		sql += 	('"' + timestamp + '"'+ ",")
		sql += ('"' + title + '"')
		sql += ("," + '"' + bodyText + '"')
		sql += ') ;'
		j += 1
		print sql
		print m
		cursor.execute(sql)
"""
sql = "CREATE TABLE BusinessNewsJapan (Timestamp VARCHAR(32),"
sql += "author VARCHAR(32) ,author_link longtext ,location longtext ,title VARCHAR(128) ,"
sql += "bodyText longtext) ENGINE=InnoDB DEFAULT CHARSET=utf8;"
cursor.execute(sql)
"""
for day in ("05282015", "05292015", "05302015", "05312015", "06012015", "06022015","06032015"):
	SelectSportsnews(day)
#connector.commit()
cursor.close()
connector.close()
