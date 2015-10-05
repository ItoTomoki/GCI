#encoding:utf8
#commenttext =  uni_text.decode('unicode_escape')
import json
import os
from ast import literal_eval
import re
import MeCab
import unicodedata
import sys
import ngram
import jcconv
from normalizer import normalize,_convert_marks,_delete_cyclic_word

argvs = sys.argv  # コマンドライン引数を格納したリストの取得
argc = len(argvs) # 引数の個数
# デバッグプリント
print argvs[1]
#print argc
#ID = '0002'
def n_gram(uni,n):
	return [uni[k:k+n] for k in range(len(uni)-n+1)]
ID = str(argvs[1])
files = os.listdir('../data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/' + ID)
thread = {}
thread[ID] = {}
index = ngram.NGram(N=2)
index3 = ngram.NGram(N=3)
kigou = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_abcdefghijklmnopqrstuvwxyz{|}~"
for nfile in files:
	filepass = ('../data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/' + ID +'/' +  str(nfile))
	#filepass = ('../tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/' + ID +'/' +  "sm20158.dat")
	f = open(filepass)
	print nfile
	lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
	Lines2 = {}
	count = 0
	for line in lines2:
		try:
			Lines2[count] = literal_eval(line)
		except:
			print line
			print count
			print nfile
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
	thread[ID][nfile] = Lines2
tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab/0.996/lib/mecab/dic/ipadic/ncnc.dic')
#commentfiles = os.listdir('comment')
for j in thread[ID].keys():
	filename = ("comment2_kai" + ID + "/" + j[0:-3] +"txt")
	#filename = ("comment2_" + ID + "/" + "sm20158." +"txt")
	fo = file(filename,'w')
	print filename
	commenttext = ''
	for i in range(0,len(thread[ID][j])):
		if i > 20000:
			print i,j
			break
		commenttext += thread[ID][j][i]["comment"]
		try:
			thread[ID][j][i]["comment"] = unicodedata.normalize("NFKC", "".join(unicodedata.normalize("NFKC", thread[ID][j][i]["comment"]).split()))
			#thread[ID][j][i]["comment"] = unicodedata.normalize('NFKC', thread[ID][j][i]["comment"])
		except:
			print "normalize Eroor"
		pluscomment = str(thread[ID][j][i]["comment"].encode('utf-8'))
		#pluscomment = jcconv.hira2kata(pluscomment) #後で追加
		"""
		pluscomment = pluscomment.replace("█", "")
		pluscomment = pluscomment.replace("□", "")
		pluscomment = pluscomment.replace("※", "")
		pluscomment = pluscomment.replace("∴", "")
		pluscomment = pluscomment.replace("*", "")
		pluscomment = pluscomment.replace("+", "")
		pluscomment = pluscomment.replace("・", "")
		pluscomment = pluscomment.replace("°", "")
		pluscomment = pluscomment.replace("w", "")
		"""
		pluscomment = pluscomment.replace("null", "")
		pluscomment = pluscomment.replace("\n", "")
		pluscomment = pluscomment.replace("\t", "")
		pluscomment = pluscomment.replace(" ", "")
		pluscomment = pluscomment.replace("　", "")
		pluscomment = pluscomment.replace("ぁ", "あ")
		pluscomment = re.sub(re.compile("[!-/:-@[-`{-~]"), '', pluscomment)
		pluscomment = normalize(pluscomment.decode("utf-8"))
		pluscomment = pluscomment.replace(u" ", "")
		pluscomment = pluscomment.replace(u"　", "")
		#さけび声対策
		pluscomment = pluscomment.replace(u"ーー",u"ー")
		if pluscomment != '':
			pluscomment = tagger.parse(pluscomment.encode("utf-8"))
			pluscomment = pluscomment.replace("\n"," ")
			pluscomment = pluscomment.replace("  "," ")
			fo.write(pluscomment)
	thread[ID][j]["comment"] = commenttext
	fo.write("\n")
	fo.close()

files = os.listdir('../data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/video')
for nfile in files[1:2]:
	#print file
	nfile = (ID + ".dat")
	filepass = ('../data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/video/' + str(nfile))
	f = open(filepass)
	lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
	f.close()
	Lines2 = {}
	count = 0
	
	for line in lines2:
		try:
			Lines2[count] = literal_eval(line)
			print Lines2[count]["video_id"], Lines2[count]["title"].decode('unicode_escape')
			thread[ID][(Lines2[count]["video_id"] + ".dat")]["title"] = Lines2[count]["title"].decode('unicode_escape')
			count += 1
		except:
			print line

