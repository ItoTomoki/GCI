#coding:utf-8

import sys
import urllib
import json

def yapi_topics():
    url = 'http://shopping.yahooapis.jp/ShoppingWebService/V1/json/categoryRanking?'
    appid = 'dj0zaiZpPTdwVVVWQ0pYYUowSCZzPWNvbnN1bWVyc2VjcmV0Jng9MjM-'
    params = urllib.urlencode(
            {'appid': appid,
             'offset':1,
             'period':'weekly',
             'generation':20,
             'gender':'male',})

    #print url + params
    response = urllib.urlopen(url + params)
    return response.read()

def do_json(s):
    data = json.loads(s)
    item_list = data["ResultSet"]["0"]["Result"]
    #print(json.dumps(item_list, sort_keys=True, indent=4)); sys.exit()
    #jsonfileを辞書型にデコード
    #print(json.dumps(data, sort_keys=True, indent=4)); sys.exit()
    #print data["ResultSet"]["0"]["Result"]["RankingData"][0]["_attributes"]["rank"]
    k = 0
    ranking = {}

    for M in item_list: # Mはkey
        try:
            ranking[int(item_list[M]["_attributes"]["rank"])] = item_list[M]["Name"]
        except:
            print ""
    print ranking.keys()
    ranking_keys = list(ranking.keys())
    ranking_keys.sort()
    print ranking_keys
    for i in ranking_keys:
        print (str(i) + u"位" + "\t" + ranking[i])

if __name__ == '__main__':
    json_str = yapi_topics()
    do_json(json_str)