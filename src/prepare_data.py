import numpy as np
import os
import json
import math
import pdb

dateIndex = 0
volumeIndex = 5
closeIndex = 6
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

def getETFData(filename, startYear=2007):
    with open(filename) as spyJsonFile:
        spyJson = json.load(spyJsonFile)
        spyArray = spyJson['dataset']['data']

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

    dataSize = len(spyArray)
    data = np.zeros([dataSize-predictAhead-lookBehind, weekCount+monthCount, dataPerDay, 1], dtype=np.float32)
    labels = np.zeros([dataSize-predictAhead-lookBehind, 2], dtype=np.int)
    
    for i in range(predictAhead, dataSize - lookBehind):
        if int(spyArray[i][dateIndex].split('-')[0]) < startYear:
            break

        outIndex = i-predictAhead # index position in the output array
        labels[outIndex][decideLabel(spyArray, i, predictAhead)] = 1
        
        for j in range(0, weekCount):
            jPos = i + j*5
            j1Pos = jPos + 5
            data[outIndex][j][0][0] = math.log(spyArray[jPos][closeIndex] / spyArray[j1Pos][closeIndex])
            data[outIndex][j][1][0] = math.log(spyArray[jPos][volumeIndex] / spyArray[j1Pos][volumeIndex])
        
        for j in range(0, monthCount):
            jPos = i + weekCount*5 + j*20
            j1Pos = jPos + 20
            data[outIndex][weekCount+j][0][0] = math.log(spyArray[jPos][closeIndex] / spyArray[j1Pos][closeIndex])
            data[outIndex][weekCount+j][1][0] = math.log(spyArray[jPos][volumeIndex] / spyArray[j1Pos][volumeIndex])

    return np.resize(data, [outIndex+1, weekCount+monthCount, dataPerDay, 1]), np.resize(labels, [outIndex+1, 2])
