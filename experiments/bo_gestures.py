import os
import reservoirpy as rpy
import numpy as np
import sys
import warnings
import sklearn.metrics
import dill
import jax
import torch

# Filter warnings and set reservoirpy verbosity
warnings.filterwarnings("ignore")
rpy.set_seed(0)
np.random.seed(0)
torch.manual_seed(0)

# Add parent directory to sys.path to import from src and gesture_recognition
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'gesture_recognition'))
data_dir = os.path.join(root_dir, 'gesture_recognition', 'dataSets')

from src.algorithms.types import EvalParams, ExperimentData
from src.algorithms.ESN_BO import ESN_BO
from src.utils import constructModel, runModel, trainModel
from src.mat_gen_v03 import bernoulli as bernoulli_v03, normal as normal_v03, uniform as uniform_v03
from gesture_recognition.Utils import createData, getData
from gesture_recognition import Evaluation

files = ['s', 'j', 'na', 'l', 'ni']

def testESN(esn, testFiles, fixed_threshold=0.4):
    testF1MaxApps = []
    testAccuracies = []
    
    _, _, trainloader, testloader = createData(inputFiles=testFiles, testFiles=testFiles, dataDir=data_dir)

    for test_inputs, test_targets in trainloader:
        inputs = test_inputs[0].numpy()
        targets = test_targets[0].numpy()
        prediction = esn.run(inputs)

        t_target = targets
        threshold = np.ones((prediction.shape[0],1))*fixed_threshold

        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction,t_target,threshold, 10)

        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, t_target, 0.5)
        testF1MaxApps.append(np.mean(sklearn.metrics.f1_score(targ_MaxApp,pred_MaxApp,average=None)))
        testAccuracies.append(np.mean(sklearn.metrics.accuracy_score(targ_MaxApp,pred_MaxApp)))
    return np.array(testF1MaxApps).mean(), np.array(testAccuracies).mean()

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
    """Evaluate a single saved fold using the dill backup."""
    save_folder = 'results/bo_gestures/global1'

    print(f"\n======================== Fold {idx+1}/5 ========================")
    testFiles = files[idx:idx+1]
    inputFiles = files[:idx] + files[idx+1:]
    train_dataset, _, train_loader, _ = createData(inputFiles, testFiles, dataDir=data_dir)
    trainX, trainY = getData(train_loader)

    fold_save_loc = os.path.join(save_folder, f'fold_{idx}')
    backup_path = os.path.join(fold_save_loc, 'bo_backup.obj')

    with open(backup_path, "rb") as f:
        bo = dill.load(f)

    model = trainModel(bo.bestModel, trainX, trainY)

    test_f1, test_acc = testESN(model, testFiles)
    print(test_f1, test_acc)
    return test_f1, test_acc

def print_saved_results():
    all_test_f1s = []
    all_test_accs = []

    for idx in [2]:
        test_f1, test_acc = print_saved_fold_results(idx)
        print(test_f1, test_acc)
        if test_f1 is not None:
            all_test_f1s.append(test_f1)
            all_test_accs.append(test_acc)

    print("\n======================== Final Results Across Folds ========================")
    if all_test_f1s:
        print(f"Average Test F1: {np.mean(all_test_f1s):.4f} (+/- {np.std(all_test_f1s):.4f})")
        print(f"Average Test Acc: {np.mean(all_test_accs):.4f} (+/- {np.std(all_test_accs):.4f})")
    else:
        print("No results to report.")

def run_bo_gestures():
    save_folder = 'results/bo_gestures/global1'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    all_test_f1s = []
    all_test_accs = []

    for idx in [2]:
        print(f"\n======================== Fold {idx+1}/5 ========================")
        inputFiles = files[:idx] + files[idx+1:]
        testFiles = files[idx:idx+1]
        validationFiles = [inputFiles[idx%4]]
        trainFiles = inputFiles[:idx%4] + inputFiles[idx%4+1:]

        print(f"Train subjects: {trainFiles}")
        print(f"Validation subject: {validationFiles}")
        print(f"Test subject: {testFiles}")

        # Load data
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
            numEvals=3,
            errorMetrics=errorMetrics,
            defaultErrors=[1.0, 0.0],
            timeout=360,
            memoryLimit=0,
            minimizeFitness=True,
            isAutoRegressive=False,
        )

        # Set input/output dims to match gesture data
        arch = {
            'nodes': [
                {'type': 'Input', 'params': {'input_dim': trainX.shape[1]}},
                {'type': 'Reservoir', 'params': {'units': 1000, 'lr': 0.9, 'sr': 0.9, 'input_connectivity': 0.25, 'rc_connectivity': 0.25, 'Win': bernoulli_v03, 'W': normal_v03, 'bias': bernoulli_v03}},
                {'type': 'Ridge', 'params': {'output_dim': trainY.shape[1], 'ridge': 8.0e-05}}
            ],
            'edges': [[0, 1], [1, 2]]
        }

        fold_save_loc = os.path.join(save_folder, f'fold_{idx}')
        if not os.path.exists(fold_save_loc):
            os.makedirs(fold_save_loc)

        # Run BO for this fold
        bo = ESN_BO(
            experimentData,
            evalParams,
            n_rand=15,
            iterations=15,
            seedModel=arch,
            n_jobs=3,
            saveLocation=os.path.join(fold_save_loc, 'bo_backup.obj'),
        )

        bo.run()

        test_f1, test_acc = print_saved_fold_results(idx)
        all_test_f1s.append(test_f1)
        all_test_accs.append(test_acc)

    print("\n======================== Final Results Across Folds ========================")
    if all_test_f1s:
        print(f"Average Test F1: {np.mean(all_test_f1s):.4f} (+/- {np.std(all_test_f1s):.4f})")
        print(f"Average Test Acc: {np.mean(all_test_accs):.4f} (+/- {np.std(all_test_accs):.4f})")
    else:
        print("No results to report.")

if __name__ == "__main__":
    run_bo_gestures()
