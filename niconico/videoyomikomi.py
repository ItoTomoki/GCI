#encoding:utf8
#commenttext =  uni_text.decode('unicode_escape')
import json
import os
from ast import literal_eval
import re
#f = open('data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/0000/sm449.dat')
files = os.listdir('data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/video')
for file in files[1:2]:
	print file
	file = "0000.dat"
	filepass = 'data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/video/' + str(file)
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
		Lines2[count] = literal_eval(line)
		print Lines2[count]["video_id"], Lines2[count]["title"].decode('unicode_escape')
		count += 1


