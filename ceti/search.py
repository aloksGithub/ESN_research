"""Sequence-classification evaluator for the existing GA + BO ESNAS engine."""

import copy
import pickle

import numpy as np
import reservoirpy as rpy
from deap import creator

from src.algorithms.ESN_GA_BO import ESNAS
from src.parallel_processing import executeParallelBatch
from ceti.models import construct_sequence_model, fit_sequences, sequence_scores


class CetiESNAS(ESNAS):
    def __init__(self, *args, frames=20, warmup=3, ridge_passes=2, seed=1234123123, **kwargs):
        super().__init__(*args, **kwargs)
        self.frames = frames
        self.warmup = warmup
        self.ridge_passes = ridge_passes
        self.run_seed = seed
        self.bo_evaluations = 1 + self.bo_init + self.bo_iter
        self.individualsPerGeneration = self.gaParams.populationSize * self.bo_evaluations

    def evaluateArchitecture(self, individual):
        data = self.experimentData
        train_x = list(data.trainX.reshape(-1, self.frames, data.trainX.shape[-1]))
        train_y = list(data.trainY.reshape(-1, self.frames, 5))
        val_x = data.valX.reshape(-1, self.frames, data.valX.shape[-1])
        attempts = []
        first_model = None
        for index in range(self.evalParams.numEvals):
            try:
                rpy.verbosity(0)
                rpy.set_seed(self.run_seed + index)
                np.random.seed(self.run_seed + index)
                model = construct_sequence_model(individual)
                fit_sequences(model, train_x, train_y, self.warmup, self.ridge_passes)
                # Retain a predetermined seed, never the luckiest validation
                # initialization. Snapshot before validation changes its state.
                if index == 0:
                    first_model = copy.deepcopy(model)
                predictions = sequence_scores(model, val_x)
                errors = [metric(data.valY, predictions) for metric in self.evalParams.errorMetrics]
                if not np.isfinite(errors).all():
                    raise ValueError("Non-finite validation metrics.")
                attempts.append(errors)
            except Exception:
                # A fragile architecture must not win through its successful
                # seeds alone. Every requested initialization must succeed.
                return individual, list(self.evalParams.defaultErrors), None
        errors = np.mean(attempts, axis=0).tolist()
        return individual, errors, first_model

    def evaluateParallel(self, population):
        # The default BO point is an additional evaluation, including when
        # bo_init=bo_iter=0. Account for it in deadlines, slicing and failures.
        old_model = self.bestModel
        self.bestModel = None
        try:
            results = executeParallelBatch(self.bo, [(individual,) for individual in population],
                self.n_jobs, self.evalParams.timeout * self.evalParams.numEvals * self.bo_evaluations)
        finally:
            self.bestModel = old_model
        evaluated, current = [], []
        for original, result in zip(population, results):
            if result is None:
                result = ([original] * self.bo_evaluations,
                          [list(self.evalParams.defaultErrors)] * self.bo_evaluations, None)
            architectures, errors, model = result
            previous_best = min((row[0] for row in self.fitnesses), default=float("inf"))
            candidate_best = min(row[0] for row in errors)
            if model is not None and candidate_best < previous_best:
                self.bestModel = model
            self.architectures.extend(architectures)
            self.fitnesses.extend(errors)
            for architecture, metrics in zip(architectures, errors):
                individual = creator.Individual(architecture)
                individual.fitness.values = (metrics[0],)
                evaluated.append(individual)
                current.append(metrics[0])
        if self.fitnesses:
            self.bestFitness = min(self.fitnesses, key=lambda errors: errors[0])
        return current, evaluated

    def run(self):
        if self.gaParams.generations == 0:
            self.population = (self.generatePopulation(self.gaParams.populationSize - len(self.seedModels)) +
                               [creator.Individual(model) for model in self.seedModels])
            _, self.population = self.evaluateParallel(self.population)
            with open(self.saveLocation, "wb") as stream:
                pickle.dump(self, stream)
            return self
        return super().run()
