#!/usr/bin/env python
# coding: utf-8

# CGI Test
import cgi
import sys
import io
import codecs

#sys.stdout = codecs.getwriter('utf_8')

html_body = u"""
<html>
<head>
<meta http-equiv="content-type" content="text/html;charset=utf-8" /> </head>
<body>
	<form name = "Form1" method="POST" action="cgitest1.py">
名前: <input type="text" size=30 name="name"><p>
アドレス: <input type="text" size=30 name="addr"><p>
<input type="submit" value="submit" name="button1"><p>
</form>
%s</body>
</html>"""
content = ""
form = cgi.FieldStorage()
form_ok = 0
k = form.getfirst("name", "")
k = cgi.escape(k)
if form.has_key("name") and form.has_key("addr") :
  form_ok = 1
if form_ok == 0 :
  content += u"Eroor"
else :
  content += u"Result\n"
  content += u"%s" % (k.decode('utf-8'))
  content += u"名前: %s" % (k)
  #content += u"addr: %s" % (u"伊藤")
print "Content-type: text/html;charset=utf-8¥n"
print type(k)
print (html_body % (content)).encode('utf-8')
