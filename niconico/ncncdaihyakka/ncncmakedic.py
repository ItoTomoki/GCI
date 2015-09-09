# -*- encoding: utf-8 -*-

import os
import csv
import re

def main():
  #入力フォルダ名
  pth = 'head'
  #出力ファイル名
  wtnme = 'ncnc.csv'
  #単語整形用の削除文字列パターン
  rmvptn = re.compile(r'(^\d[1,2]月\d[1,2]日$)|((\(|（).+(\)|）)$)') #月日タグとタグ後ろのジャンル名は削除

  with open(wtnme,'wb') as wtfh:
    wt = csv.writer(wtfh)
    fnmes = os.listdir(pth)
    for fnme in fnmes:
      try:
        print fnme
        with open(os.path.join(pth,fnme),'rb') as rdfh:
          rd = csv.reader(rdfh)
          for row in rd:
            if row[3]=='a':
              wrd = rmvptn.sub('',row[1]).lower()
              if(0 < len(wrd)):
                wt.writerow(
                  [wrd,'0','0',int(max(-32768.0, (6000 - 200 *(len(wrd)**1.3)))),'名詞','一般','*','*','*','*',wrd,row[2],row[2],'ニコニコ大百科']
                )
      except:
        print "Error"
        print fnme
        continue

if __name__ == '__main__':
  main()