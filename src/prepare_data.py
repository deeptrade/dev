import numpy as np
import os
import json
import math
import pdb
import constants as const
from utils import dataSum
from utils import decideLabel

# The index as presented in spy.json file
dateIndex = 0
volumeIndex = 5
closeIndex = 6

dataPerDay = const.DATA_PER_DAY

def getETFData(filename, startYear=2007):
    with open(filename) as spyJsonFile:
        spyJson = json.load(spyJsonFile)
        spyArray = spyJson['dataset']['data']

    # in our calculation, we always treat 5 trading day as 1 week, and 20 trading day as 1 month
    # currently look back 10 x 20-day data point, and 10 x 5-day data point)
    # look ahead 20 day.
    predictAhead = const.PREDICT_DAYS_AHEAD
    weekCount = const.WEEK_COUNT
    lookBehind = weekCount * 5

    dataSize = len(spyArray)
    data = np.zeros([dataSize-predictAhead-lookBehind, weekCount, 1, dataPerDay], dtype=np.float32)
    labels = np.zeros([dataSize-predictAhead-lookBehind, const.NUM_CLASSES], dtype=np.int)
    
    for i in range(predictAhead, dataSize - lookBehind - 5):
        if int(spyArray[i][dateIndex].split('-')[0]) < startYear:
            break

        outIndex = i-predictAhead # index position in the output array
        labels[outIndex][decideLabel(spyArray, i, predictAhead, closeIndex)] = 1
        
        for j in range(0, weekCount):
            jPos = i + j*5
            j1Pos = jPos + 5
            data[outIndex][j][0][0] = math.log(dataSum(spyArray, jPos, 5, closeIndex) / dataSum(spyArray, j1Pos, 5, closeIndex))
            data[outIndex][j][0][1] = math.log(dataSum(spyArray, jPos, 5, volumeIndex) / dataSum(spyArray, j1Pos, 5, volumeIndex))

    return np.resize(data, [outIndex+1, weekCount, 1, dataPerDay]), np.resize(labels, [outIndex+1, const.NUM_CLASSES])
