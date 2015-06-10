# -*- coding: utf-8 -*-
import pandas.io.data as web
import matplotlib.pyplot as plt
import datetime

# 取得する日の範囲を指定する
start = datetime.datetime(2013, 5, 1)
end = datetime.datetime(2013, 5, 31)

# Yahoo ファイナンスから、 ^N225 (日経平均株価指数) を
# とってくる。
f = web.DataReader('^N225', 'yahoo', start, end)

plt.title('Nikkei 255 from 2013.5.1 to 2013.5.31')

# fill_between でその日の最高値と最低値をプロットする
plt.fill_between(f.index, f['Low'], f['High'], color="b", alpha=0.2)

# plot で、始値をプロットする。
# 自動的に Index が Date になっているので、横軸が時間になる。

f['Open'].plot()
print f[:10]
plt.show()