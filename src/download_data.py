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
import pdb
import constants as const
from utils import dataSum
from utils import decideLabel
from utils import getDailyData

BASEURL="https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?qopts.columns=ticker,date,adj_close,adj_volume"

# parameter regarding the raw data format
tickerIndex = 0
dateIndex = 1
closeIndex = 2
volumeIndex = 3

# parameter used when parsing and serializing data
weekCount = const.WEEK_COUNT
dayCount = weekCount*5
dataPerDay = const.DATA_PER_DAY

def getStockData(filename, untilYear=2007):
    with open(filename) as spyJsonFile:
        spyJson = json.load(spyJsonFile)
        spyArray = spyJson['datatable']['data']

        # clean the data
        if spyArray == None:
            print("discarding stock")
            return [], []

        reverseArray = []
        skipped = 0
        for d in reversed(spyArray):
            if int(d[dateIndex].split('-')[0]) >= untilYear:
                continue

            if d[closeIndex] == 0 or d[volumeIndex] == 0 or d[closeIndex] == None or d[volumeIndex] == None:
                print("found zero value {}".format(d))
                skipped += 1
                if skipped > 10:
                    print("discarding stock")
                    return [], []
                continue
            reverseArray.append(d)
        spyArray = reverseArray

    dataSize = len(spyArray)
    if dataSize < 400:
        # just ignore the ones with a short history
        print("discarding stock")
        return [], []

    # data, labels = getWeeklyData(spyArray, closeIndex, volumeIndex)

    data, labels = getDailyData(spyArray, 0, closeIndex, 0, 0, volumeIndex)
    return np.reshape(data, [-1, dayCount*dataPerDay]), labels

currentDir = os.path.dirname(os.path.realpath(__file__))

parser = argparse.ArgumentParser(description='downloading data from quandl')
parser.add_argument("-i", "--input", help="Set input file name.")
parser.add_argument("-o", "--output", help="Set output directory name.")
parser.add_argument("-k", "--key", help="Set the quandl api key")
parser.add_argument("-u", "--until", help="Only train on data that's before this year (default 2007)")

args = parser.parse_args()
if args.input == None:
    args.input = "{}/../data/sp500.csv".format(currentDir)
if args.output == None:
    args.output = "{}/../data/sp500".format(currentDir)
if args.until == None:
    args.until = 2007

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
        print("\nprocessing {}".format(filename))
        data, labels = getStockData("{}/{}".format(args.output, filename), untilYear=int(args.until))
        if len(data) == 0:
            continue
        
        labelsum = sum(labels)
        print("... {} entries, buy {} hold {} sell {}".format(len(data), labelsum[2], labelsum[1], labelsum[0]))
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
            
