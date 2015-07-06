#encoding:utf8
import mysql.connector
import MySQLdb
import numpy as np
import codecs
import sys
#sys.stdout = codecs.getwriter("utf8")(sys.stdout)
#sys.stdin = codecs.getreader("sutf8")(sys.stdin)
connector = MySQLdb.connect(host="localhost", db="Restaurant2", user="root",passwd="", charset="utf8")
cursor = connector.cursor()
import csv
def writedata(FileName):
	fileName = ("RCdata/" + FileName + ".csv")
	f = open(fileName, 'r')

	sql = "CREATE TABLE " + FileName  + " ( "
	reader = csv.reader(f)
	header = next(reader)
	#sql = "CREATE TABLE chefmozaccepts ("
	for column in header:
		sql += (str(column) + " " + "LONGTEXT" + ",")
	sql = sql[0:-1]
	sql += ") ENGINE=InnoDB DEFAULT CHARSET=utf8 ;"
	try:
		cursor.execute(sql)
	except:
		print sql
	for row in reader:
		sql = "INSERT INTO " + FileName + " ("
		for column in header:
			sql += (str(column) + ",")
		sql = sql[0:-1]
		sql += ") VALUES("
		i = 0
		for column in header:
			sql += ("'" + str(row[i]) + "'" + ",")
			i += 1
		sql = sql[0:-1]
		sql += ");"
		try:
			cursor.execute(sql)
		except:
			print sql#だめなやつはしょうがないので後で手作業で入れる
	f.close()
for FileName in ("geoplaces2","userprofile", "rating_final"):
	writedata(FileName)
connector.commit()
cursor.close()
connector.close()
