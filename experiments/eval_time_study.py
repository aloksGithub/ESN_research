import time
import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from NAS.utils import evaluateArchitecture, generateRandomArchitectureOld
from NAS.parallel_processing import executeParallelBatch
from utils import getDataMGS


def main():
    trainX, trainY, valX, valY, testX, testY = getDataMGS()

    def eval_func(architecture):
        start = time.time()
        evaluateArchitecture(architecture, trainX, trainY, valX, valY, 3)
        print(time.time() - start)
    
    architectures = []
    for _ in range(20):
        architectures.append(generateRandomArchitectureOld(trainX.shape[1], trainY.shape[1], trainX, trainY, 756, 60))
    
    executeParallelBatch(eval_func, architectures, 20, 600 * 3)

if __name__ == "__main__":
    main()