#!/usr/bin/env python
#coding: utf-8

import cgi
from datetime import datetime
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = 'utf-8')

html_body="""
<html><meta charset="utf-8"><body>
%s
</body></html>"""

content=u'こい'

form=cgi.FieldStorage()
title=form.getvalue('title','')
print title
memo = form.getvalue('memo','')
print "Content-type: text/html; charset=utf-8\n"
print (html_body % content).encode('utf-8')
print (html_body % title).encode('utf8')
print (html_body % memo).encode('utf-8')

