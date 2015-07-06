import csv

f = open('some.csv', 'r')

reader = csv.reader(f)
header = next(reader)
for row in reader:
    print row

f.close()