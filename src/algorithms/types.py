class ExperimentData:
    def __init__(self, trainX, trainY, valX, valY, testX, testY):
        self.trainX = trainX
        self.trainY = trainY
        self.valX = valX
        self.valY = valY
        self.testX = testX
        self.testY = testY

class GAParams:
    def __init__(
        self,
        generations,
        populationSize,
        crossoverProbability,
        mutationProbability,
        eliteSize,
        stagnationReset,
        earlyStop=None,
    ):
        self.generations = generations
        self.populationSize = populationSize
        self.crossoverProbability = crossoverProbability
        self.mutationProbability = mutationProbability
        self.eliteSize = eliteSize
        self.stagnationReset = stagnationReset
        self.earlyStop = earlyStop

class EvalParams:
    def __init__(
        self,
        numEvals,
        errorMetrics,
        defaultErrors,
        isAutoRegressive,
        timeout,
        memoryLimit,
        minimizeFitness,
    ):
        self.numEvals = numEvals
        self.errorMetrics = errorMetrics
        self.defaultErrors = defaultErrors
        self.isAutoRegressive = isAutoRegressive
        self.timeout = timeout
        self.memoryLimit = memoryLimit
        self.minimizeFitness = minimizeFitness

class ModelParams:
    def __init__(self, num_nodes_range=(2, 4)):
        self.num_nodes_range = num_nodes_range