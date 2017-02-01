import numpy as np
import logging
import os
import math
import tensorflow as tf
from prepare_data import getData

logger = logging.getLogger("dt")


logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s %(threadName)s: %(message)s")
logger.setLevel(logging.DEBUG)

data, labels = getData()


print("done")