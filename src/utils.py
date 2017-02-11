import constants as const 

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
