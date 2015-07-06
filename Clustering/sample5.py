#coding:utf-8

import csv
import math

f = open('iris.csv','rb')
data = []
c = csv.reader(f)
for row in c:
    data.append(row)
f.close()

vector = [] #データをベクトルに入れる
num  = len(data)
for i in range(1,num):
    vector.append(data[i])
    data[i].pop(0)

vector_num = len(vector) #類似度の行列を作る
mat = []
dist = 0

def make_mat(vector):
    sub_mat = []
    for i in range(1,len(vector)):
        for j in range(i):  # ユークリッド距離の計算
            dist = math.sqrt((float(vector[i][0])-float(vector[j][0]))**2+(float(vector[i][1])-float(vector[j][1]))**2+(float(vector[i][2])-float(vector[j][2]))**2+(float(vector[i][3])-float(vector[j][3]))**2)
            sub_mat.append(dist)
            dist = 0
        mat.append(sub_mat)
        sub_mat = []
    return mat

def dd(list1,list2):
    dist = math.sqrt((float(list1[0])-float(list2[0]))**2+(float(list1[1])-float(list2[1]))**2+(float(list1[2])-float(list2[2]))**2+(float(list1[3])-float(list2[3]))**2)
    return dist

def min(list):  #最小値検索をする関数
    min = list[0]
    for i in range(len(list)):
        if min >= list[i]:
            min = list[i]
    return min

def min_num(list):  #最小値検索をする関数
    min_number = 0
    min = list[0]
    for i in range(len(list)-1):
        if min >= list[i]:
            min_number = i
            min = list[i]
    return min_number

def mat_min(matrix):
    m = matrix[0][0]
    row = 0
    for i in range(len(matrix)):
        if m > min(matrix[i]):
            m = min(matrix[i])
            row = i
    col = min_num(matrix[row])
    return m,row,row+1,col #データに対応するのはrow+1,colである。

def mean(list1,list2):   #データの平均値リストを返す
    mm = [(float(list1[0])+float(list2[0]))*1.0/2,(float(list1[1])+float(list2[1]))*1.0/2,(float(list1[2])+float(list2[2]))*1.0/2,(float(list1[3])+float(list2[3]))*1.0/2,list1[4]+"/"+list2[4]]
    return mm

def large(a,b):
    if a > b:
        return [b,a]
    if b > a:
        return [a,b]

def re_mat(mat,vector,num): #最小値をグルーピングをし、再構成を行う関数
    row = num[2]
    col = num[3]
    dist = mat_min(mat)[0]
    ll = large(row,col)
    del mat[ll[0]]
    del mat[ll[1]-1]
    mat.append(vector)
    store = [row,col,dist]
    return mat,store

for i in range(len(vector)-3):
    mat = []
    mat = make_mat(vector)
    mmmat = mat_min(mat)
    ave = mean(vector[mmmat[2]],vector[mmmat[3]])
    rr = re_mat(vector,ave,mmmat)
    print rr
    #    print len(rr[0])
    mat = rr
print "*************"
print rr[0][0][4]
print "*************"
print rr[0][1][4]
print "*************"
print rr[0][2][4]
print "*************"