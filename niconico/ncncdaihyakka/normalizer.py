# coding: utf-8

import re
import unicodedata

from tornado import escape

# url, htmltag(<>で囲まれた要素)
_re_remove = re.compile(r"(http(s)?://[\w.\-/:#?=&;\\%~\+]+|<[^>]*?>)")
# 空文字
_re_escape = re.compile(r"&[\w]+;")


def normalize(sentence):
    """
    コメントを正規化する
    :param str sentence: 正規化するコメント
    :return: 正規化されたコメント
    :rtype: str
    """

    dst = escape.xhtml_unescape(sentence)

    if _re_escape.findall(dst):
        return ""

    # か゛(u'\u304b\u309b')" -> が(u'\u304b \u3099')
    #  -> が(u'\u304b\u3099') -> が(u'\u304c')
    #dst = unicodedata.normalize("NFKC", "".join(unicodedata.normalize("NFKC", dst).split()))
    dst = dst.lower()
    dst = "".join(dst.split())
    try:
        dst = _convert_marks(dst)
    except:
        print "convertError"
    dst = _re_remove.sub("", dst)
    dst = _delete_cyclic_word(dst)

    return dst

# 記号変換テーブル
_convert_mark_dic = {"〜": "-", "~": "-"}
_re_convert_mark = re.compile(r"[〜~]")


def _convert_marks(sentence):
    """
    特殊な記号変換
    _conver_mark_dicテーブルを使って意図的な変換をかける
    :param str sentence: 返還前の文字列
    :return: 返還後の文字列
    :rtype: str
    """
    return _re_convert_mark.sub(lambda x: _convert_mark_dic[x.group()], sentence)


def _delete_cyclic_word(sentence):
    """
    繰り返しワードの除去
    同一文字は2個まで繰り返しok
    それ以外の繰り返しは正規化される
    Exsample:
        abab -> ab
        foo -> foo
    :param str sentence: 除去したい文字列
    :return: 除去後の文字列
    :rtype: str
    """
    i = len(sentence) // 2
    while 0 < i:
        for j in range(len(sentence) - 2 * i + 1):
            x, y, z = j, j + i, j + 2 * i

            if sentence[x:y] == sentence[y:z]:
                if i == 1:
                    if z != len(sentence) and sentence[z] == sentence[y]:
                        sentence = sentence[:y] + sentence[z:]
                        i = len(sentence) // 2
                        break
                else:
                    sentence = sentence[:y] + sentence[z:]
                    i = len(sentence) // 2
                    break
        else:
            i -= 1

    return sentence