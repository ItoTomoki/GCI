#coding: utf-8
import urllib2
import lxml.html

#変数htmlには上記のHTMLがstrで代入されているとします。
html = urllib2.urlopen("http://www.reuters.com")
dom = lxml.html.fromstring(html)
k = unicode(dom, 'utf-8')
#変数title, timestamp, author, author_link, bodyにそれぞれタイトル、投稿日時、著者、著者のリンク、記事本文が代入されます。
title = k.xpath('//h1')[0].text
timestamp = dom.xpath('//*[@id="articleInfo"]//*[@class="timestamp"]')[0].text
author = dom.xpath   ('//*[@id="articleInfo"]//*[@class="author"]/a')[0].text
author_link = dom.xpath   ('//*[@id="articleInfo"]//*[@class="author"]/a')[0].attrib['href']
body = dom.xpath('//*[@id="articleText"]')[0].text