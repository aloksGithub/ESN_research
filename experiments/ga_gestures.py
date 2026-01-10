import reservoirpy as rpy
import numpy as np
import sys
import os
import warnings
import traceback
import sklearn.metrics

# Filter warnings and set reservoirpy verbosity
warnings.filterwarnings("ignore")
rpy.verbosity(0)

# Add parent directory to sys.path to import from src and gesture_recognition
current_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'gesture_recognition'))

from src.algorithms.ESN_GA import ESN_GA, EvalParams, ExperimentData, GAParams, ModelParams
from src.utils import runModel, trainModel
from gesture_recognition.Utils import createData, getData
from gesture_recognition import Evaluation

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
    This is used during GA search (validation).
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

def run_ga_gestures():
    files = ['s', 'j', 'na', 'l', 'ni']
    save_folder = 'results/ga_gestures/global1'
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
            numEvals=3,
            errorMetrics=errorMetrics,
            defaultErrors=[1.0, 0.0],
            timeout=60,
            memoryLimit=756,
            minimizeFitness=True,
            isAutoRegressive=False,
        )
        
        # GA parameters matching ga.py (larger population and more generations)
        gaParams = GAParams(
            generations=40,
            populationSize=20,
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

        # Run GA for this fold
        ga = ESN_GA(
            experimentData,
            evalParams,
            gaParams,
            modelParams,
            n_jobs=4,
            saveLocation=os.path.join(fold_save_loc, 'ga_backup.obj'),
        )
        
        ga.run()
        
        # Final evaluation: Following optimizer_global1's testModel logic
        # 1. Re-train best model on combined train+validation data
        # 2. Evaluate over 10 trials
        best_model = ga.bestModel
        if best_model is not None:
            print("Re-training best model on combined train+validation data...")
            combinedX = np.concatenate([trainX, np.concatenate(valX)])
            combinedY = np.concatenate([trainY, np.concatenate(valY)])
            
            best_model = trainModel(best_model, combinedX, combinedY)
            
            preds = [runModel(best_model, seq) for seq in testX]
            
            # Accumulate per-sequence scores
            for target, prediction in zip(testY, preds):
                threshold = np.ones((prediction.shape[0], 1)) * 0.4
                t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction, target, threshold, 10)
                pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, target, 0.5)
                
                test_f1 = np.mean(sklearn.metrics.f1_score(targ_MaxApp, pred_MaxApp, average=None))
                test_acc = np.mean(sklearn.metrics.accuracy_score(targ_MaxApp, pred_MaxApp))
            
            all_test_f1s.append(test_f1)
            all_test_accs.append(test_acc)
        else:
            print(f"Fold {idx+1} failed to find a valid model.")

    print("\n======================== Final Results Across Folds ========================")
    if all_test_f1s:
        print(f"Average Test F1: {np.mean(all_test_f1s):.4f} (+/- {np.std(all_test_f1s):.4f})")
        print(f"Average Test Acc: {np.mean(all_test_accs):.4f} (+/- {np.std(all_test_accs):.4f})")
    else:
        print("No results to report.")

if __name__ == "__main__":
    run_ga_gestures()
