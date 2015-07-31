#encoding:utf8
#commenttext =  uni_text.decode('unicode_escape')
import json
import os
from ast import literal_eval
import re
import MeCab
#f = open('data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/0000/sm449.dat')
files = os.listdir('data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/0000')
thread = {}
thread["0000"] = {}
#for file in files
for file in files[-50:-1]:
	print file
	#thread["0000"][file] = {}
	filepass = 'data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/0000/' + str(file)
	f = open(filepass)
	lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
	data1 = f.read()  # ファイル終端まで全て読んだデータを返す
	f.close()
	#print type(data1) # 文字列データ
	Lines2 = {}
	#lines1 = data1.split('n') # 改行で区切る(改行文字そのものは戻り値のデータには含まれない)
	#print type(lines2)
	count = 0
	for line in lines2:
		try:
			Lines2[count] = literal_eval(line)
		except:
			print line
			print count
			print file
			line = line.replace('null', '"null"')
			print line
			try:
				Lines2[count] = literal_eval(line)
			except:
				continue			
                        
		try:
			Lines2[count]['comment'] = Lines2[count]['comment'].decode('unicode_escape')
		except:
			try:
				#print ("Eroor1" + Lines2[count]['comment'])
				Lines2[count]['comment'] = Lines2[count]['comment'][0:-1]
			except:
				print ("Eroor2" + line)
		#print Lines2[count]['comment']
		count += 1
	thread["0000"][file] = Lines2
tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/wikipedia-keyword.dic')

for j in thread["0000"].keys():
	commenttext = ''
	#for i in range(0,len(thread["0000"]["sm9922.dat"])):
	for i in range(0,len(thread["0000"][j])):
		commenttext += thread["0000"][j][i]["comment"]
		#print thread["0000"]["sm9922.dat"][i]["comment"]
	thread["0000"][j]["comment"] = commenttext
commentwakati = tagger.parse(str(commenttext.encode('utf-8')))
#print commentwakati.decode("utf-8")
print thread["0000"][thread["0000"].keys()[8]]["comment"]
print tagger.parse(str(thread["0000"][thread["0000"].keys()[8]]["comment"].encode('utf-8')))






