#!/usr/bin/python
# -*- coding: utf-8 -*-
import datetime
today = str(datetime.date.today())
print 'Content-type: text/html\n'
print """
<!DOCTYPE html>
<html>
<head><title>CGIスクリプト</title></head>
<body>
これはサーバの実行結果として生成されたHTMLです<br>
今日は%sです
</body></html>
"""%today
