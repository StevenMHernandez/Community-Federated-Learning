#Imports
import csv

def read(filePath):
    data = []
    with open(filePath) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for line in reader:
            data.append(line)
    return data

def readWithDelimiter(filePath, delimiter):
    data = []
    with open(filePath) as csvFile:
        reader = csv.reader(csvFile, delimiter)
        for line in reader:
            data.append(line)
    return data

