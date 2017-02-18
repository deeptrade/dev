import constants as const
import numpy as np
import tensorflow as tf
import pdb

dateIndex = const.DATE_INDEX
openIndex = const.OPEN_INDEX
highIndex = const.HIGH_INDEX
lowIndex = const.LOW_INDEX
volumeIndex = const.VOLUME_INDEX
closeIndex = const.CLOSE_INDEX

def dataSum(spyArray, index, count, columnIndex):
    s = 0
    for i in range(index, index+count):
        s += spyArray[i][columnIndex]
    return float(s)

# given the data and range, decide whether it's a buy (1) or sell (0)
# the data is always sorted inversely by time.
def decideLabel(spyArray, i, predictAhead, closeIndex):
    # simply use close price
    return int(spyArray[i][closeIndex] < spyArray[i-predictAhead][closeIndex])
    '''
    # This policy gives a buy signal if for predictAhead number of days, the
    # average is always higher than the current price.
    current = 0
    future = 0
    decision = []
    for j in range(1, predictAhead+1):
        current += spyArray[i][closeIndex]
        future += spyArray[i-j][closeIndex]
        decision.append(int(future > current))
    
    # the goal is to use a rule that's not very arbitrary
    total = sum(decision)
    if total == predictAhead:
        return 2
    elif total == 0:
        return 0
    return 1
    '''

    '''
    avg = dataSum(spyArray, i-predictAhead, predictAhead, closeIndex) / predictAhead
    if currentClose > avg:
        return 1
    return 0

    if avg >= high:
        return 2
    elif avg <= low:
        return 0
    else:
        return 1
    '''

# Get weekly average data from json array
def getWeeklyData(spyArray):
    # in our calculation, we always treat 5 trading day as 1 week, and 20 trading day as 1 month
    # currently look back 64 weeks, look ahead 20 days. For each week data point, we take the average
    # for that week.
    weekCount = const.WEEK_COUNT
    dataPerDay = const.DATA_PER_DAY
    predictAhead = const.PREDICT_DAYS_AHEAD
    lookBehind = weekCount * 5
    dataSize = len(spyArray)

    data = np.zeros([dataSize-predictAhead-lookBehind, weekCount, 1, dataPerDay], dtype=np.float32)
    labels = np.zeros([dataSize-predictAhead-lookBehind, const.NUM_CLASSES], dtype=np.int)
    
    for i in range(predictAhead, dataSize - lookBehind - 5):
        outIndex = i-predictAhead # index position in the output array
        labels[outIndex][decideLabel(spyArray, i, predictAhead, closeIndex)] = 1
        
        for j in range(0, weekCount):
            jPos = i + j*5
            j1Pos = jPos + 5
            data[outIndex][j][0][0] = (dataSum(spyArray, jPos, 5, closeIndex) / dataSum(spyArray, j1Pos, 5, closeIndex)) - 1.0
            data[outIndex][j][0][1] = (dataSum(spyArray, jPos, 5, volumeIndex) / dataSum(spyArray, j1Pos, 5, volumeIndex)) - 1.0
        
        # debug print
        '''
        print("{} {} {}".format(spyArray[i][dateIndex], spyArray[i][closeIndex], labels[outIndex]))
        for j in range(0, weekCount):
            print("diff {}".format(data[outIndex][j][0]))
        print("")
        '''
    return np.reshape(data, [-1, weekCount*dataPerDay]), labels

def getDailyData(spyArray):
    # in our calculation, we always treat 5 trading day as 1 week, and 20 trading day as 1 month
    # currently look back 10 x 20-day data point, and 10 x 5-day data point)
    # look ahead 20 day.
    predictAhead = const.PREDICT_DAYS_AHEAD
    dayCount = const.WEEK_COUNT * 5
    lookBehind = dayCount
    dataPerDay = const.DATA_PER_DAY

    dataSize = len(spyArray)
    data = np.zeros([dataSize-predictAhead-lookBehind, dayCount, 1, dataPerDay], dtype=np.float32)
    labels = np.zeros([dataSize-predictAhead-lookBehind, const.NUM_CLASSES], dtype=np.int)
    
    for i in range(predictAhead, dataSize - lookBehind - 5):
        outIndex = i-predictAhead # index position in the output array
        labels[outIndex][decideLabel(spyArray, i, predictAhead, closeIndex)] = 1
        
        for j in range(0, dayCount):
            data[outIndex][j][0][0] = spyArray[i+j][closeIndex]/spyArray[i+j+1][closeIndex] - 1.0
            data[outIndex][j][0][1] = spyArray[i+j][volumeIndex]/spyArray[i+j+1][volumeIndex] - 1.0
            data[outIndex][j][0][2] = spyArray[i+j][openIndex]/spyArray[i+j+1][openIndex] - 1.0
            data[outIndex][j][0][3] = spyArray[i+j][highIndex]/spyArray[i+j+1][highIndex] - 1.0
            data[outIndex][j][0][4] = spyArray[i+j][lowIndex]/spyArray[i+j+1][lowIndex] - 1.0

    return np.resize(data, [outIndex+1, dayCount, 1, dataPerDay]), np.resize(labels, [outIndex+1, const.NUM_CLASSES])

def getDailyImageData(spyArray):
    predictAhead = const.PREDICT_DAYS_AHEAD
    dayCount = const.WEEK_COUNT * 5
    dataPerDay = const.DATA_PER_DAY

    dataSize = len(spyArray)
    data = np.zeros([dataSize-predictAhead, dayCount, 1, dataPerDay], dtype=np.float32)
    labels = np.zeros([dataSize-predictAhead, const.NUM_CLASSES], dtype=np.int)
    
    with tf.Session():
        for i in range(predictAhead, dataSize-dayCount):
            outIndex = i-predictAhead # index position in the output array
            labels[outIndex][decideLabel(spyArray, i, predictAhead, closeIndex)] = 1
            factor = dataSum(spyArray, i, dayCount, closeIndex) / dataSum(spyArray, i, dayCount, volumeIndex)
            
            for j in range(0, dayCount):
                data[outIndex][j][0][0] = spyArray[i+j][closeIndex]
                data[outIndex][j][0][1] = spyArray[i+j][openIndex]
                data[outIndex][j][0][2] = spyArray[i+j][highIndex]
                data[outIndex][j][0][3] = spyArray[i+j][lowIndex]
                data[outIndex][j][0][4] = spyArray[i+j][volumeIndex] * factor

    return np.resize(data, [outIndex+1, dayCount, 1, dataPerDay]), np.resize(labels, [outIndex+1, const.NUM_CLASSES])

def getDailySquareImageData(spyArray):
    # Get the daily data, and the same number of moving averages to form a square image
    predictAhead = const.PREDICT_DAYS_AHEAD
    dayCount = const.WEEK_COUNT * 5
    lookBehind = dayCount*2
    dataPerDay = const.DATA_PER_DAY

    dataSize = len(spyArray)
    #data = np.zeros([dataSize-predictAhead-lookBehind, dayCount, dayCount, dataPerDay], dtype=np.float32)
    data = []
    labels = []
    
    for i in range(predictAhead, dataSize - lookBehind - 5):
        outIndex = i-predictAhead # index position in the output array
        label = [0, 0]
        label[decideLabel(spyArray, i, predictAhead, closeIndex)] = 1
        labels.append(label)
        
        image = np.zeros([dayCount, dayCount, dataPerDay], dtype=np.float32)
        # can be further optimized, since day 2 is just day 1's data shifted, with one additional day added.
        for j in range(0, dayCount):
            # j dimension is the number of simple moving averages days
            avgLen = j+1
            image[j][0][0] = dataSum(spyArray, i, avgLen, closeIndex) / avgLen
            image[j][0][1] = dataSum(spyArray, i, avgLen, openIndex) / avgLen
            image[j][0][2] = dataSum(spyArray, i, avgLen, highIndex) / avgLen
            image[j][0][3] = dataSum(spyArray, i, avgLen, lowIndex) / avgLen
            for k in range(1, dayCount):
                # k dimension is the day 
                image[j][k][0] = image[j][k-1][0] - image[0][k-1][0] / avgLen + spyArray[i+k+j][closeIndex] / avgLen
                image[j][k][1] = image[j][k-1][1] - image[0][k-1][1] / avgLen + spyArray[i+k+j][openIndex] / avgLen
                image[j][k][2] = image[j][k-1][2] - image[0][k-1][2] / avgLen + spyArray[i+k+j][highIndex] / avgLen
                image[j][k][3] = image[j][k-1][3] - image[0][k-1][3] / avgLen + spyArray[i+k+j][lowIndex] / avgLen

        data.append(image)
        print("processed {}".format(i))
    
    return np.resize(data, [outIndex+1, dayCount, dayCount, dataPerDay]), np.resize(labels, [outIndex+1, const.NUM_CLASSES])
