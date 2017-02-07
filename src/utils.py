
def dataSum(spyArray, index, count, columnIndex):
    s = 0
    for i in range(index, index+count):
        s += spyArray[i][columnIndex]
    return s

# given the data and range, decide whether it's a buy (1) or sell (0)
# the data is always sorted inversely by time.
def decideLabel(spyArray, i, predictAhead, closeIndex):
    currentClose = spyArray[i][closeIndex]
    total = dataSum(spyArray, i-predictAhead, predictAhead, closeIndex)
        
    # this model issues a buy signal if the average is higher than the current price. 
    if (total / predictAhead) > currentClose:
        return 1
    return 0
