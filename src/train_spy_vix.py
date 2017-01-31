#!/usr/local/bin/python3

import numpy as np
import logging
import os
import json
import math

logger = logging.getLogger("dt")
volumeIndex = 5
closeIndex = 6

if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(threadName)s: %(message)s")
    logger.setLevel(logging.DEBUG)

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

    # currently look back 220 data points (10 20-day data point, and 4 5-day data point)
    # look ahead 5 day.
    predictAhead = 5
    weekCount = 4 # 5 data points is counted as 1 week
    monthCount = 10 # 20 data points is counted as 1 month
    lookBehind = weekCount * 5 + (monthCount+1) * 20

    dataSize = min(len(spyArray), len(vixArray))
    data = np.empty([dataSize-predictAhead-lookBehind, weekCount+monthCount, 3], dtype=np.float32)
    labels = np.empty(dataSize-predictAhead-lookBehind, dtype=np.int)
    
    for i in range(predictAhead, dataSize - lookBehind):
        outIndex = i-predictAhead
        
        currentClose = spyArray[i][closeIndex]
        minIndex = 0
        minValue = currentClose
        maxIndex = 0
        maxValue = currentClose
        for j in range(1, predictAhead):
            p = spyArray[i-j][closeIndex]
            if p > maxValue:
                maxValue = p
                maxIndex = j
            elif p < minValue:
                minValue = p
                minIndex = j
        
        if minIndex == 0:
            labels[outIndex] = 1
        elif maxIndex == 0:
            labels[outIndex] = 0
        else:
            labels[outIndex] = (minIndex > maxIndex)

        for j in range(0, weekCount):
            data[outIndex][j][0] = math.log(spyArray[i + j*5][closeIndex] / spyArray[i + (j+1)*5][closeIndex])
            data[outIndex][j][1] = math.log(spyArray[i + j*5][volumeIndex] / spyArray[i + (j+1)*5][volumeIndex])
            data[outIndex][j][2] = math.log(vixArray[i + j*5][closeIndex] / 20)
        
        for j in range(0, monthCount):
            data[outIndex][weekCount+j][0] = math.log(spyArray[i + weekCount*5 + j*20][closeIndex] / spyArray[i + weekCount*5 + (j+1)*20][closeIndex])
            data[outIndex][weekCount+j][1] = math.log(spyArray[i + weekCount*5 + j*20][volumeIndex] / spyArray[i + weekCount*5 + (j+1)*20][volumeIndex])
            data[outIndex][weekCount+j][2] = math.log(vixArray[i + weekCount*5 + j*20][closeIndex] / 20)

        print("done")