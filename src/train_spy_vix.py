#!/usr/local/bin/python3

import numpy as np
import logging
import os
import json
import math
import tensorflow as tf

logger = logging.getLogger("dt")
volumeIndex = 5
closeIndex = 6

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

def getData():
    currentDir = os.path.dirname(os.path.realpath(__file__))
    with open(currentDir+'/../data/spy.json') as spyJsonFile:
        spyJson = json.load(spyJsonFile)
        spyArray = spyJson['dataset']['data']

    with open(currentDir+'/../data/vix.json') as vixJsonFile:
        vixJson = json.load(vixJsonFile)
        vixArray = vixJson['dataset']['data']
    
    assert(spyArray[0][0] == vixArray[0][0])
    '''
    print(spyArray[0])
    print(spyArray[0][volumeIndex])
    print(vixArray[0])
    print(vixArray[0][volumeIndex])
    '''

    # in our calculation, we always treat 5 trading day as 1 week, and 20 trading day as 1 month
    # currently look back 10 x 20-day data point, and 10 x 5-day data point)
    # look ahead 20 day.
    predictAhead = 20
    weekCount = 10
    monthCount = 10
    lookBehind = weekCount * 5 + monthCount * 20

    dataSize = min(len(spyArray), len(vixArray))
    data = np.empty([dataSize-predictAhead-lookBehind, weekCount+monthCount, 3], dtype=np.float32)
    labels = np.empty(dataSize-predictAhead-lookBehind, dtype=np.int)
    
    for i in range(predictAhead, dataSize - lookBehind):
        outIndex = i-predictAhead # index position in the output array
        labels[outIndex] = decideLabel(spyArray, i, predictAhead)
        
        for j in range(0, weekCount):
            jPos = i + j*5
            j1Pos = jPos + 5
            data[outIndex][j][0] = math.log(spyArray[jPos][closeIndex] / spyArray[j1Pos][closeIndex])
            data[outIndex][j][1] = math.log(spyArray[jPos][volumeIndex] / spyArray[j1Pos][volumeIndex])
            data[outIndex][j][2] = math.log(vixArray[jPos][closeIndex] / vixArray[j1Pos][closeIndex])
        
        for j in range(0, monthCount):
            jPos = i + weekCount*5 + j*20
            j1Pos = jPos + 20
            data[outIndex][weekCount+j][0] = math.log(spyArray[jPos][closeIndex] / spyArray[j1Pos][closeIndex])
            data[outIndex][weekCount+j][1] = math.log(spyArray[jPos][volumeIndex] / spyArray[j1Pos][volumeIndex])
            data[outIndex][weekCount+j][2] = math.log(vixArray[jPos][closeIndex] / 20)

    return data, labels


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(threadName)s: %(message)s")
    logger.setLevel(logging.DEBUG)

    data, labels = getData()


    print("done")