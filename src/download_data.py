'''
Download s&p500 stock data, using symbols described in the input file.
The input file should be a csv with the first column being the stock symbols. 
'''

import argparse
import urllib
import os
import sys
import tensorflow as tf
import numpy as np
import json
import math

BASEURL="https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?qopts.columns=ticker,date,adj_close,adj_volume"

volumeIndex = 3
closeIndex = 2
dataPerDay = 2 # close price, volume

# given the data and range, decide whether it's a buy (1) or sell (0)
# the data is always sorted inversely by time.
def decideLabel(spyArray, i, predictAhead):
    currentClose = spyArray[i][closeIndex]
    total = 0.0
    for j in range(1, predictAhead+1):
        total += spyArray[i-j][closeIndex]
        
    # this model issues a buy signal if the average is higher than the current price. 
    if (total / predictAhead) > currentClose:
        return 1
    return 0

def getStockData(filename):
    with open(filename) as spyJsonFile:
        spyJson = json.load(spyJsonFile)
        spyArray = spyJson['datatable']['data']
        spyArray.reverse()
    if spyArray == None:
        return [], []

    # in our calculation, we always treat 5 trading day as 1 week, and 20 trading day as 1 month
    # currently look back 10 x 20-day data point, and 10 x 5-day data point)
    # look ahead 20 day.
    predictAhead = 20
    weekCount = 10
    monthCount = 10
    lookBehind = weekCount * 5 + monthCount * 20

    dataSize = len(spyArray)
    if dataSize < 400:
        # just ignore the ones with a short history
        return [], []

    data = np.zeros([dataSize-predictAhead-lookBehind, weekCount+monthCount, dataPerDay, 1], dtype=np.float32)
    labels = np.zeros([dataSize-predictAhead-lookBehind, 2], dtype=np.int)
    
    for i in range(predictAhead, dataSize - lookBehind):
        outIndex = i-predictAhead # index position in the output array
        labels[outIndex][decideLabel(spyArray, i, predictAhead)] = 1
        
        for j in range(0, weekCount):
            jPos = i + j*5
            j1Pos = jPos + 5
            if spyArray[j1Pos][closeIndex] == 0 or spyArray[j1Pos][volumeIndex] == 0 or \
              spyArray[j1Pos][closeIndex] == None or spyArray[j1Pos][volumeIndex] == None:
                print("found zero value at index {} value {}".format(j1Pos, spyArray[j1Pos]))
                return [], []
            data[outIndex][j][0][0] = math.log(spyArray[jPos][closeIndex] / spyArray[j1Pos][closeIndex])
            data[outIndex][j][1][0] = math.log(spyArray[jPos][volumeIndex] / spyArray[j1Pos][volumeIndex])
        
        for j in range(0, monthCount):
            jPos = i + weekCount*5 + j*20
            j1Pos = jPos + 20
            if spyArray[j1Pos][closeIndex] == 0 or spyArray[j1Pos][volumeIndex] == 0 or \
              spyArray[j1Pos][closeIndex] == None or spyArray[j1Pos][volumeIndex] == None:
                print("found zero value at index {} value {}".format(j1Pos, spyArray[j1Pos]))
                return [], []
            data[outIndex][weekCount+j][0][0] = math.log(spyArray[jPos][closeIndex] / spyArray[j1Pos][closeIndex])
            data[outIndex][weekCount+j][1][0] = math.log(spyArray[jPos][volumeIndex] / spyArray[j1Pos][volumeIndex])

    return np.reshape(data, [-1, (weekCount+monthCount)*dataPerDay]), labels

currentDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='downloading data from quandl')
parser.add_argument("-i", "--input", help="Set input file name.")
parser.add_argument("-o", "--output", help="Set output directory name.")
parser.add_argument("-k", "--key", help="Set the quandl api key")

args = parser.parse_args()
if args.input == None:
    args.input = "{}/../data/sp500.csv".format(currentDir)
if args.output == None:
    args.output = "{}/../data/sp500".format(currentDir)

# download again if the output directory doesn't have any json files
found = False
for filename in os.listdir(args.output):
    if filename.endswith(".json"): 
        found = True
        break

if not found:
    if args.key == None:
        print("argements not specified, run with -h to see the help")
        exit(0)

    with open(args.input, 'r') as fin:
        fin.readline() # skip the first line, which is the header
        for line in fin.readlines():
            ar = line.split(',')

            print("reading {}\n".format(ar[0]))
            urlstr = "{}&api_key={}&ticker={}".format(BASEURL, args.key, ar[0])
            urllib.urlretrieve(urlstr, "{}/{}.json".format(args.output, ar[0]))

writer = tf.python_io.TFRecordWriter("{}/all.bin".format(args.output))
# from the json files generate training data.
for filename in os.listdir(args.output):
    if filename.endswith(".json"): 
        print("processing {}".format(filename))
        data, labels = getStockData("{}/{}".format(args.output, filename))
        if len(data) == 0:
            continue

        assert(len(data) == len(labels))
        shuffle_indices = np.random.permutation(np.arange(len(data)))
        data = data[shuffle_indices]
        labels = labels[shuffle_indices]

        data = data.tolist()
        labels = labels.tolist()
        for i in range(0, len(data)):
            example = tf.train.Example(
                features=tf.train.Features(feature={
                    # A Feature contains one of either a int64_list,
                    # float_list, or bytes_list
                    'label': tf.train.Feature(
                        int64_list=tf.train.Int64List(value=labels[i])),
                    'data': tf.train.Feature(
                        float_list=tf.train.FloatList(value=data[i])),
            }))
            serialized = example.SerializeToString()
            writer.write(serialized)
            
