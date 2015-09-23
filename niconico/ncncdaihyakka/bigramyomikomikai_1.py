#encoding:utf8
import os
from ast import literal_eval
import re
import MeCab
import unicodedata
import sys
import ngram


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
kigou = "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_abcdefghijklmnopqrstuvwxyz{|}~"
index = ngram.NGram(N=2)
for nfile in files:
	filepass = ('../data/tcserv.nii.ac.jp/access/tomokiitoupcfax@gmail.com/832c5b059b15f647/nicocomm/data/thread/' + ID +'/' +  str(nfile))
	f = open(filepass)
	lines2 = f.readlines() # 1行毎にファイル終端まで全て読む(改行文字も含まれる)
	#data1 = f.read()  # ファイル終端まで全て読んだデータを返す
	f.close()
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
#tagger = MeCab.Tagger( '-Owakati -u /usr/local/Cellar/mecab-ipadic/2.7.0-20070801/lib/mecab/dic/ipadic/ncnc.dic')
#commentfiles = os.listdir('comment')

for j in thread[ID].keys():
	filename = ("commentbigram_kai" + ID + "/" + j[0:-3] +"txt")
	#filename = ("comment2bigram_kai" + ID + "/" + j[0:-3] +"txt")
	#filename = ("comment2_" + ID + "/" + j[0:-3] +"txt")
	fo = file(filename,'w')
	print filename
	commenttext = ''
	for i in range(0,len(thread[ID][j])):
		if i > 20000:
			print i,j
			break
		commenttext += thread[ID][j][i]["comment"]
		try:
			thread[ID][j][i]["comment"] = unicodedata.normalize('NFKC', thread[ID][j][i]["comment"])
		except:
			print "normalize Eroor"
		pluscomment = str(thread[ID][j][i]["comment"].encode('utf-8'))
		#pluscomment = jcconv.hira2kata(pluscomment) #後で追加
		pluscomment = pluscomment.replace("█", "")
		pluscomment = pluscomment.replace("□", "")
		pluscomment = pluscomment.replace("※", "")
		pluscomment = pluscomment.replace("∴", "")
		pluscomment = pluscomment.replace("*", "")
		pluscomment = pluscomment.replace("+", "")
		pluscomment = pluscomment.replace("・", "")
		pluscomment = pluscomment.replace("°", "")
		pluscomment = pluscomment.replace("w", "")
		pluscomment = pluscomment.replace("null", "")
		pluscomment = pluscomment.replace("\n", "")
		pluscomment = pluscomment.replace("\t", "")
		pluscomment = pluscomment.replace(" ", "")
		pluscomment = pluscomment.replace("　", "")
		pluscomment = pluscomment.replace("ぁ", "あ")
		pluscomment = re.sub(re.compile("[!-/:-@[-`{-~]"), '', pluscomment)
		#さけび声対策
		pluscommentlist = list(index.ngrams(index.pad(pluscomment.decode("utf-8"))))
		text = ''
		word1 = ''
		word2  =''
		for word in pluscommentlist:
			if word == u"ーー":
				continue
			if word != word1:
				text += word[0]
			word1 = word
		if len(text) > 0:
			pluscomment =  text#[1::]
		#繰り返し対策
		""""
		pluscommentlist = list(index.ngrams(index.pad(pluscomment)))
		text = ''
		word1 = ''
		word2  =''
		for n in range(0,len(pluscommentlist)):
			word = pluscommentlist[n]
			if n >= 2:
				if word == pluscommentlist[n-2]:
					text += (" " + word[0])
					continue
			text += word[0]
		if len(text) > 0:
			pluscomment =  text
		"""
		pluscommentlist = n_gram(pluscomment,3)
		text = ''
		word1 = ''
		word2  =''
		#くりかえす対策/消す
		for n in range(0,len(pluscommentlist)):
			word = pluscommentlist[n]
			if n >= 3:
				if pluscommentlist[n] == pluscommentlist[n-3]:
					continue
			if n >= 4:
				if pluscommentlist[n] == pluscommentlist[n-4]:
					continue
			if n >= 5:
				if pluscommentlist[n] == pluscommentlist[n-5]:
					continue
			if n != (len(pluscommentlist)-1):
				text += word[0]
			else:
				text += word
		if len(text) > 0:
			pluscomment = text
		pluscomment = pluscomment.replace("$","")
		if pluscomment != '':
			for text in list(index.ngrams(index.pad(pluscomment))):
				fo.write(text.encode("utf-8") + " ")
			fo.write("\n")
	thread[ID][j]["comment"] = commenttext
	fo.write("\n")
	fo.close()


