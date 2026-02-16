import os
import reservoirpy as rpy
import numpy as np
import sys
import warnings
import traceback
import sklearn.metrics
import dill
from deap import base, creator
import jax

# Filter warnings and set reservoirpy verbosity
warnings.filterwarnings("ignore")
rpy.set_seed(42)
np.random.seed(42)

# Add parent directory to sys.path to import from src and gesture_recognition
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'gesture_recognition'))

from src.algorithms.ESN_GA_BO import EvalParams, ExperimentData, GAParams, ModelParams, ESNAS
from src.utils import constructModel, runModel, trainModel
from gesture_recognition.Utils import createData, getData
from gesture_recognition import Evaluation

files = ['s', 'j', 'na', 'l', 'ni']

def getSequenceData(loader):
    """Extracts individual sequences from the DataLoader instead of concatenating them."""
    x = []
    y = []
    for inputs, targets in loader:
        x.append(inputs[0].numpy())
        y.append(targets[0].numpy())
    return x, y

def gesture_f1_metric(y_true, y_pred):
    """
    Mimics testESN from Train.py: averages F1 scores across sequences.
    This is used during ESNAS search (validation).
    """
    if not isinstance(y_true, list):
        # Fallback if somehow concatenated arrays are passed
        y_true = [y_true]
        y_pred = [y_pred]
    
    f1_scores = []
    for target, prediction in zip(y_true, y_pred):
        # Following testESN: threshold=0.4, minLength=10
        threshold = np.ones((prediction.shape[0], 1)) * 0.4
        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction, target, threshold, 10)
        
        # Following testESN: threshold=0.5 for calcInputSegmentSeries
        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, target, 0.5)
        
        # Calculate per-sequence macro F1
        score = np.mean(sklearn.metrics.f1_score(targ_MaxApp, pred_MaxApp, average=None))
        f1_scores.append(score)
    
    # Return 1 - average F1 (to minimize)
    return 1.0 - np.mean(f1_scores)

def gesture_acc_metric(y_true, y_pred):
    """Mimics testESN from Train.py: averages Accuracy across sequences."""
    if not isinstance(y_true, list):
        y_true = [y_true]
        y_pred = [y_pred]
    
    acc_scores = []
    for target, prediction in zip(y_true, y_pred):
        threshold = np.ones((prediction.shape[0], 1)) * 0.4
        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction, target, threshold, 10)
        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, target, 0.5)
        
        acc = np.mean(sklearn.metrics.accuracy_score(targ_MaxApp, pred_MaxApp))
        acc_scores.append(acc)
    
    return np.mean(acc_scores)

def print_saved_fold_results(idx):
    """Evaluate a single saved fold using the jax.export artifact."""
    save_folder = 'results/esnas_gestures/global1'

    print(f"\n======================== Fold {idx+1}/5 ========================")
    inputFiles = files[:idx] + files[idx+1:]
    testFiles = files[idx:idx+1]
    trainFiles = inputFiles[:idx%4] + inputFiles[idx%4+1:]

    # Seed RNGs so test data is identical across runs
    # (createData adds random noise and shuffles segments)
    import random as _random
    _random.seed(42 + idx)
    np.random.seed(42 + idx)

    # Load data
    data_dir = os.path.join(root_dir, 'gesture_recognition', 'dataSets')

    # Test data (List of sequences)
    _, test_dataset, _, test_loader = createData(trainFiles, testFiles, dataDir=data_dir)
    testX, testY = getSequenceData(test_loader)
    testX, testY = testX[0], testY[0]

    fold_save_loc = os.path.join(save_folder, f'fold_{idx}')

    # Register deap creator types before loading (they're created dynamically)
    if not hasattr(creator, "Fitness"):
        creator.create("Fitness", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", dict, fitness=creator.Fitness)

    # Load dill backup to get the save path, then load jax.export for inference
    with open(os.path.join(fold_save_loc, 'esnas_backup.obj'), "rb") as f:
        ga = dill.load(f)
    export_path = os.path.splitext(ga.saveLocation)[0] + "_forward.jax"
    predict = ESNAS.loadPredict(export_path)

    preds = predict(testX)

    threshold = np.ones((preds.shape[0], 1)) * 0.4
    t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(preds, testY, threshold, 10)
    pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, testY, 0.5)

    test_f1 = np.mean(sklearn.metrics.f1_score(targ_MaxApp, pred_MaxApp, average=None))
    test_acc = np.mean(sklearn.metrics.accuracy_score(targ_MaxApp, pred_MaxApp))
    print(f"F1: {test_f1:.4f}, Acc: {test_acc:.4f}")

    return test_f1, test_acc


def print_saved_results():
    all_test_f1s = []
    all_test_accs = []

    for idx in range(5):
        test_f1, test_acc = print_saved_fold_results(idx)
        if test_f1 is not None:
            all_test_f1s.append(test_f1)
            all_test_accs.append(test_acc)

    print("\n======================== Final Results Across Folds ========================")
    if all_test_f1s:
        print(f"Average Test F1: {np.mean(all_test_f1s):.4f} (+/- {np.std(all_test_f1s):.4f})")
        print(f"Average Test Acc: {np.mean(all_test_accs):.4f} (+/- {np.std(all_test_accs):.4f})")
    else:
        print("No results to report.")

def run_esnas_gestures():
    save_folder = 'results/esnas_gestures/global1'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_test_f1s = []
    all_test_accs = []

    for idx in range(5):
        print(f"\n======================== Fold {idx+1}/5 ========================")
        inputFiles = files[:idx] + files[idx+1:]
        testFiles = files[idx:idx+1]
        validationFiles = [inputFiles[idx%4]]
        trainFiles = inputFiles[:idx%4] + inputFiles[idx%4+1:]
        
        print(f"Train subjects: {trainFiles}")
        print(f"Validation subject: {validationFiles}")
        print(f"Test subject: {testFiles}")

        # Load data
        data_dir = os.path.join(root_dir, 'gesture_recognition', 'dataSets')
        
        # Training data (Concatenated for fitting)
        train_dataset, _, train_loader, _ = createData(trainFiles, testFiles, dataDir=data_dir)
        trainX, trainY = getData(train_loader)
        
        # Validation data (List of sequences for per-sequence averaging)
        _, val_dataset, _, val_loader = createData(trainFiles, validationFiles, dataDir=data_dir)
        valX, valY = getSequenceData(val_loader)
        valX, valY = valX[0], valY[0]
        
        # Test data (List of sequences)
        _, test_dataset, _, test_loader = createData(trainFiles, testFiles, dataDir=data_dir)
        testX, testY = getSequenceData(test_loader)
        testX, testY = testX[0], testY[0]

        experimentData = ExperimentData(trainX, trainY, valX, valY, testX, testY)

        # Define error metrics following testESN behavior
        errorMetrics = [gesture_f1_metric, gesture_acc_metric]
        
        evalParams = EvalParams(
            numEvals=1,
            errorMetrics=errorMetrics,
            defaultErrors=[1.0, 0.0],
            timeout=20,
            memoryLimit=756,
            minimizeFitness=True,
            isAutoRegressive=False,
        )
        
        gaParams = GAParams(
            generations=2,
            populationSize=8,
            crossoverProbability=0.7,
            mutationProbability=0.2,
            eliteSize=1,
            stagnationReset=5,
        )
        
        modelParams = ModelParams(
            num_nodes_range=(2, 4),
        )

        fold_save_loc = os.path.join(save_folder, f'fold_{idx}')
        if not os.path.exists(fold_save_loc):
            os.makedirs(fold_save_loc)

        # Run ESNAS for this fold
        ga = ESNAS(
            experimentData,
            evalParams,
            gaParams,
            modelParams,
            n_jobs=4,
            saveLocation=os.path.join(fold_save_loc, 'esnas_backup.obj'),
            bo_init=0,
            bo_iter=5
        )
        
        ga.run()

        test_f1, test_acc = print_saved_fold_results(idx)
        if test_f1 is not None:
            all_test_f1s.append(test_f1)
            all_test_accs.append(test_acc)

    print("\n======================== Final Results Across Folds ========================")
    if all_test_f1s:
        print(f"Average Test F1: {np.mean(all_test_f1s):.4f} (+/- {np.std(all_test_f1s):.4f})")
        print(f"Average Test Acc: {np.mean(all_test_accs):.4f} (+/- {np.std(all_test_accs):.4f})")
    else:
        print("No results to report.")

if __name__ == "__main__":
    print_saved_results()
