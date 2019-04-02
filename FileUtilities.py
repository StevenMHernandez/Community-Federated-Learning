#Imports
import csv

def loadData(fileName):
    data = []
    with open(fileName) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for line in reader:
            data.append(line)
    return data

def loadDataWithDelimiter(fileName, delimiter):
    data = []
    with open(fileName) as csvFile:
        reader = csv.reader(csvFile, delimiter)
        for line in reader:
            data.append(line)
    return data

# Function: Creates a CSV file output containing the information from the provided array.
def exportResults(data, fileName):
    with open('output_' + fileName) as outputFile:
        csvWriter = csv.writer(outputFile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            csvWriter.writerow(row)
    return True

# Function: Creates a CSV file output containing the information from the
#           provided array using provided delimiter.
def exportResultsWithDelimiter(data, fileName, delimiter):
    with open('output_' + fileName) as outputFile:
        csvWriter = csv.writer(outputFile, delimiter=delimiter, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in data:
            csvWriter.writerow(row)
    return True

# Function: Splits the provided data equally among communities.
def distibuteData(data, numCommunities):
    communityData = []
    dataIndex = 0
    #Initialize communityData to be appropriate size for numCommunities
    for i in range(0, numCommunities):
        communityData.append([])

    #What if the number of communities is greater than amount of remaining data?
    #Distribute data to each community
    while dataIndex < len(data):
        for i in range(0,numCommunities):
            if dataIndex < len(data):
                communityData[i].append(data[dataIndex])
                dataIndex += 1
    return communityData

# Function: Splits the provided data uniformly, respective to labels, among the communities.
# Note: Assumes that final column in data is the data's label!
def lazyUniformDistributeData(data, numCommunities):
    if len(data) < 2:
        return []
    numLabels = 0
    numFeatures = len(data[0])
    #Count all of the unique labels and initialize the dictionary
    labels = {}
    labelsCounter = 0
    for row in data:
        if row[numFeatures - 1] not in labels.keys():
            labels[row[numFeatures - 1]] = []
    #Place all the data with the appropriate label
    for row in data:
        labels[row[numFeatures - 1]].append(row)
    #Split!
    communityData = []
    for i in range(0, numCommunities):
        communityData.append([])
    for label in labels:
        distributedLabelData = distibuteData(labels[label], numCommunities)
        #print(distributedLabelData)
        #Need an extra for loop here to add the distributedLabelData to communityData
        for i in range(0,len(distributedLabelData)):
            communityData[i].append(distributedLabelData[i])
    return communityData




