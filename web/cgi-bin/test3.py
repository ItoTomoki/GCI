#!/usr/bin/env python
# -*- coding: utf-8 -*-
import cgi
import os
import sys
print u"""“Content-type: text/html\n”"""
print u"""“<html lang=\”ja\”>”"""
print u"""“<head> <meta http-equiv=\”Content-Type\” content=\”text/html; charset=UTF-8\”></head>”"""
print u"""“””<body><h1>Python CGI Test</h1><h3> with MeCab</h3>”””"""
print “<p>日本語を入力してください。MeCabの解析結果を出力します</p>”
print “””<form method=”POST”><textarea name=”TextArea” rows=”10″ cols=”60″></textarea>”””
print “””<input type=”submit” name=”subm1″ value=”submit”></form>”””
form = cgi.FieldStorage()
name = form[“TextArea”].value
print “あなたの入力は:<br>\”” + name +”\”\n<br><br>”
print “<table>”
print “<tr><th>表層</th><th>品詞</th><th>分類1</th><th>分類2</th><th>分類3</th>”
print “<th>活用形</th><th>活用型</th><th>原型</th><th>読み</th><th>発音<th></tr>”
print “</table>”
print “</body></html>”