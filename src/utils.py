import constants as const 

def dataSum(spyArray, index, count, columnIndex):
    s = 0
    for i in range(index, index+count):
        s += spyArray[i][columnIndex]
    return s

# given the data and range, decide whether it's a buy (1) or sell (0)
# the data is always sorted inversely by time.
def decideLabel(spyArray, i, predictAhead, closeIndex):
    currentClose = spyArray[i][closeIndex]
    #high = currentClose * (1 + const.BUY_THRESHOLD)
    #low = currentClose * (1 - const.SELL_THRESHOLD)

    avg = dataSum(spyArray, i-predictAhead, predictAhead, closeIndex) / predictAhead
    if currentClose > avg:
        return 1
    return 0
    
    '''
    if avg >= high:
        return 2
    elif avg <= low:
        return 0
    else:
        return 1
    '''
