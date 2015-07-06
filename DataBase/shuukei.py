#encoding:utf8
import mysql.connector
import MySQLdb
import numpy as np
import codecs
import sys
#sys.stdout = codecs.getwriter("utf8")(sys.stdout)
#sys.stdin = codecs.getreader("sutf8")(sys.stdin)
connector = MySQLdb.connect(host="localhost", db="Restaourant", user="root",passwd="", charset="utf8")
cursor = connector.cursor()

SELECT count(*) FROM userprofile;
SELECT count(*) FROM geoplaces2;
SELECT count(*) FROM chefmozcuisine WHERE Rcuisine = "Japanese";
SELECT Rcuisine,count(*) FROM chefmozcuisine GROUP BY Rcuisine;

SELECT name, AVG(rating) 
FROM geoplaces2 
LEFT JOIN rating_final
ON geoplaces2.placeID = rating_final.placeID
GROUP BY geoplaces2.placeID;

SELECT chefmozcuisine.Rcuisine, AVG(rating) as average 
FROM rating_final 
LEFT JOIN userprofile ON userprofile.userID = rating_final.userID 
LEFT JOIN chefmozcuisine ON chefmozcuisine.placeID = rating_final.placeID
WHERE userprofile.dress_preference = "formal" and chefmozcuisine.Rcuisine is not NULL
GROUP BY chefmozcuisine.Rcuisine
ORDER BY average DESC;

SELECT chefmozcuisine.Rcuisine, AVG(rating) as average 
FROM rating_final 
LEFT JOIN userprofile ON userprofile.userID = rating_final.userID 
LEFT JOIN chefmozcuisine ON chefmozcuisine.placeID = rating_final.placeID
WHERE userprofile.dress_preference  <> "formal" and chefmozcuisine.Rcuisine is not NULL
GROUP BY chefmozcuisine.Rcuisine
ORDER BY average DESC;

SELECT chefmozparking.parking_lot, AVG(rating) as average
FROM rating_final 
LEFT JOIN chefmozparking ON rating_final.placeID = chefmozparking.placeID
GROUP BY chefmozparking.parking_lot 
ORDER BY average DESC

SELECT usercuisine.Rcuisine, AVG(rating) as average
FROM rating_final
LEFT JOIN usercuisine ON rating_final.UserID = usercuisine.UserID
RIGHT JOIN userprofile ON usercuisine.UserID = userprofile.UserID
WHERE userprofile.smoker = "False"
GROUP BY usercuisine.Rcuisine
ORDER BY average DESC;

