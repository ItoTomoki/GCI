#coding:utf-8

import sys
import urllib
import json
import urllib2
import xml.etree.ElementTree as etree

def yapi_topics():
    url = 'http://shopping.yahooapis.jp/ShoppingWebService/V1/itemSearch?'
    appid = 'dj0zaiZpPTdwVVVWQ0pYYUowSCZzPWNvbnN1bWVyc2VjcmV0Jng9MjM-'
    resp = urllib2.urlopen(url + 'appid=%s'%appid + '&category_id=%d&sort=-sold'%635).read()
    output = {}
    tree = etree.fromstring(resp)
    
    for e in tree[-1]:
        #print e
        #output[e.attrib.keys()] = e.attrib.values()
        #print output
        print e.attrib.values(), e.attrib.keys()
        
if __name__ == '__main__':
    json_str = yapi_topics()
    #do_json(json_str)