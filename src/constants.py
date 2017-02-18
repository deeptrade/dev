
WEEK_COUNT=8           # How many weeks to include in one data point
DATA_PER_DAY=5          # open, close, high, low, volume
PREDICT_DAYS_AHEAD=1    # Predict the average price for the next how many days

NUM_CLASSES=2           # The number of classes in your classification.
BUY_THRESHOLD=0.0       # percentage higher than the current value to trigger a buy
SELL_THRESHOLD=0.0      # percentage lower than the current value to trigger a sell

# The index as presented in json files
DATE_INDEX = 0
OPEN_INDEX = 1
HIGH_INDEX = 2
LOW_INDEX = 3
VOLUME_INDEX = 5
CLOSE_INDEX = 6
