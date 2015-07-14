#!d:\Python25\python.exe
#coding:utf-8
'''

'''
import  cgi
import  cgitb; cgitb.enable() #エラー発生時にブラウザ上に表示させる、デバック時のみ利用がおすすめ

print "Content-Type: text/html"
print 
html = '''<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
</head>
<body>'''
print html
print "Hello World"
print "日本語<br>"
print "cgi.test()結果"

fs = cgi.FieldStorage()

#チェックボックス
chkbox = fs.getlist("chkbox")#つねにリストにして返す、値がない場合は空のリスト、getfirstは最初の値だけ返すが順番は保障しない
for item in chkbox:
  print "name:chkbox=",  item,"<br>"
#ラジオボタン
radio = fs.getlist("radio")
for item in radio:
  print "name:radio=",  item,"<br>"
#テキスト
#text = fs["text"].value.decode("shift-jis")#同一名のフィールドの場合はインスタンスのリストになる
text = fs.getvalue("text").decode("utf8")
print "name:text=" , text.encode("utf8"),"<br>"
#メモ
memo = fs.getvalue("memo").decode("utf8")
print "name:memo=" , memo.encode("utf-8"),"<br>"

#ボタン
btn = fs.getlist('btn')#同名ボタンが存在しているパターン
for item in btn:
  print "name:btn=",  item,"<br>"

print '<a href="index.html">index</a>'
print "</body>"
print "</html>"