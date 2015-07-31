# coding:utf-8

import urllib2
import json
import sys
import codecs
sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
def nico_search(query, service=['video'],
                search=['title', 'description', 'tags'],
                join=['cmsid', 'title', 'view_counter'],
                issuer='nico_api'):
    req = urllib2.Request('http://api.search.nicovideo.jp/api/snapshot/')
    req.add_header('content-type', 'application/json')

    data = json.dumps({'query': query, 'service': service, 'search': search,'join': join, 'issuer': issuer})
    res = unicode(urllib2.urlopen(req, data).read(), 'utf-8')
    return res

if __name__ == '__main__':
    res = nico_search('初音ミク')
    print res
