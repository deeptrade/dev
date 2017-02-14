import constants as const
import numpy as np

def dataSum(spyArray, index, count, columnIndex):
    s = 0
    for i in range(index, index+count):
        s += spyArray[i][columnIndex]
    return s

# given the data and range, decide whether it's a buy (1) or sell (0)
# the data is always sorted inversely by time.
def decideLabel(spyArray, i, predictAhead, closeIndex):
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
def getWeeklyData(spyArray, closeIndex, volumeIndex):
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

def getDailyData(spyArray, openIndex, closeIndex, highIndex, lowIndex, volumeIndex):
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
            '''
            data[outIndex][j][0][2] = spyArray[i+j][openIndex]/spyArray[i+j+1][openIndex] - 1.0
            data[outIndex][j][0][3] = spyArray[i+j][highIndex]/spyArray[i+j+1][highIndex] - 1.0
            data[outIndex][j][0][4] = spyArray[i+j][lowIndex]/spyArray[i+j+1][lowIndex] - 1.0
            '''

    return np.resize(data, [outIndex+1, dayCount, 1, dataPerDay]), np.resize(labels, [outIndex+1, const.NUM_CLASSES])
