from deap import creator
import pickle
import time

from ..algorithms.GA_Base import GA_Base
from ..algorithms.types import EvalParams, ExperimentData, GAParams, ModelParams
from ..parallel_processing import executeParallelBatch

class ESN_GA(GA_Base):
    """Genetic algorithm to obtain an optimized ESN architecture for a dataset"""

    def __init__(
        self,
        experimentData: ExperimentData,
        evalParams: EvalParams,
        gaParams: GAParams,
        modelParams: ModelParams,
        seedModels=[],
        n_jobs=1,
        saveModels=False,
        saveLocation=None,
    ):
        super().__init__(
            experimentData,
            evalParams,
            gaParams,
            modelParams,
            seedModels,
            n_jobs,
            saveModels,
            saveLocation,
        )

    def evaluateParallel(self, population):
        print("Evaluating population")
        results = executeParallelBatch(
            (self.evaluateArchitecture),
            [(individual,) for individual in population],
            self.n_jobs,
            self.evalParams.timeout * self.evalParams.numEvals,
        )
        for i in range(len(results)):
            if results[i] is None:
                results[i] = (population[i], self.evalParams.defaultErrors, None)

        for result in results:
            ind, errors, model = result
            self.fitnesses.append(errors)
            self.architectures.append(ind)
            if (
                errors[0] <= min([elem[0] for elem in self.fitnesses])
                or len(self.fitnesses) == 0
            ):
                self.bestModel = model
            if self.saveModels:
                self.models.append(model)
            ind.fitness.values = (errors[0],)
        return [performanceData[1][0] for performanceData in results]

    def generationRun(self, gen: int):
        startTime = time.time()
        print("=======================Generation {}=======================".format(gen))
        self.generationsSinceImprovement += 1
        offspring = self.generateOffspring(
            list(map(self.toolbox.clone, self.population))
        )

        # Evaluate offspring
        offSpringFitnesses = self.evaluateParallel(offspring)
        if (
            self.evalParams.minimizeFitness
            and min(offSpringFitnesses) < self.prevFitness
            or not self.evalParams.minimizeFitness
            and max(offSpringFitnesses) > self.prevFitness
        ):
            self.prevFitness = min(offSpringFitnesses)
            self.generationsSinceImprovement = 0

        if self.generationsSinceImprovement >= self.gaParams.stagnationReset:
            print("Resetting population due to stagnation")

            self.prevFitness = self.evalParams.defaultErrors[0]
            newRandomPopulation = self.generatePopulation(
                self.gaParams.populationSize - 1
            )
            self.evaluateParallel(newRandomPopulation)
            self.population[:] = (
                self.toolbox.selectBest(self.population, 1) + newRandomPopulation
            )
            self.modelGenerationIndices.append(gen)
        else:
            self.population[:] = offspring

        objective = [errors[0] for errors in self.fitnesses]
        bestIndex = (
            objective.index(min(objective))
            if self.evalParams.minimizeFitness
            else objective.index(max(objective))
        )
        self.bestFitness = self.fitnesses[bestIndex]
        numFailures = 0
        for index, fitness in enumerate(
            self.fitnesses[-self.gaParams.populationSize :]
        ):
            if fitness[0] == self.evalParams.defaultErrors[0]:
                # print(self.architectures[-self.populationSize:][index])
                numFailures += 1
        print("Best so far:", self.bestFitness)
        print(
            "Failure rate: {}%".format(100 * numFailures / self.gaParams.populationSize)
        )
        self.generationTimes.append(time.time() - startTime)
        print("Time taken:", time.time() - startTime)

        file = open(self.saveLocation, "wb")
        pickle.dump(self, file)

    def run(self):
        startTime = time.time()
        random_population = self.generatePopulation(
            self.gaParams.populationSize - len(self.seedModels)
        )
        seed_population = [
            creator.Individual(individual) for individual in self.seedModels
        ]
        self.population = seed_population + random_population

        self.evaluateParallel(self.population)
        self.modelGenerationIndices.append(0)
        endTime = time.time()
        self.generationTimes.append(endTime - startTime)

        for gen in range(self.generation, self.gaParams.generations + 1):
            self.generationRun(gen)
            if self.gaParams.earlyStop is not None:
                if self.evalParams.minimizeFitness:
                    if self.bestFitness[0] <= self.gaParams.earlyStop:
                        break
                else:
                    if self.bestFitness[0] >= self.gaParams.earlyStop:
                        break

        file = open(self.saveLocation, "rb")
        return pickle.load(file)
