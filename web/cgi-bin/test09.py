#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
テキスト入力フィールドに入力された文字を取得する
'''
html = '''Content-Type: text/html

<html>
<head>
  <title>u"テキスト入力フィールドに入力された文字を取得する"</title>
</head>
<body>
<h1>テキスト入力フィールドに入力された文字を取得する</h1>
<p>入力された文字は、「%s」です。</p>
<form action="test09.py" method="post">
  <input type="text" name="text" />
  <input type="submit" />
</form>
</body>
</html>
'''

import cgi
f = cgi.FieldStorage()
txt = f.getfirst('text', '')
print html % cgi.escape(txt)