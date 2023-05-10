# Topology and parameter optimization

## Evolutionary strategy for simultaneous optimization of parameters, topology and reservoir weights in Echo State Networks 

### Objective
use an evolutionary strategy in order to search the best values of the reservoir global parameters, the best topology and the reservoir weight sizes simultaneously

### Evolutionary algorithm
- Dataset creation: Time series is normalized and split into training/validation (75%) and testing (25%) sets.

- Population initialization: A random initial population of 120 individuals is created, each represented by a vector 's' with ESN information.

- Topology creation: For each 's', a topology is created with corresponding elements. All topologies have one input and one output.

- Readout training and performance evaluation: A readout function is trained, and reservoir performance is evaluated using 10-fold cross-validation.

- Fitness computation: Fitness value is calculated based on performance in training and validation sets.

- New populations: Algorithm creates new populations through elitism, crossover, and mutation operations.

- Stopping condition: Algorithm stops after 10 generations.

- Test performance: Best ESN performance is computed for the test set (25% of the data).

### Datasets

They used two wind speed time series from the SONDA project, collected at Triunfo and Belo Jardim in Brazil. The performance of their models can be summarized in the following tables:

| Dataset   | NMSE  | MSE(%) | MAPE(%) | MAE (m/s) |
|-----------|-------|--------|---------|-----------|
| Triunfo - Training   | 0.0803 | 0.1379 | 7.07    | 0.86    |
| Triunfo - Validation | 0.1697 | 0.1074 | 6.93    | 0.84    |
| Triunfo - Test       | 0.1886 | 0.1071 | 7.06    | 0.84    |
| Triunfo - Persistence | 0.7056 | 0.4006 | 14.43   | 1.70    |

| Dataset       | NMSE  | MSE(%) | MAPE(%) | MAE (m/s) |
|---------------|-------|--------|---------|-----------|
| Belo Jardim - Training   | 0.2544 | 0.5032 | 16.70   | 0.69    |
| Belo Jardim - Validation | 0.2751 | 0.4747 | 16.56   | 0.68    |
| Belo Jardim - Test       | 0.3016 | 0.5519 | 17.08   | 0.73    |
| Belo Jardim - Persistence | 0.8608 | 1.5749 | 30.16   | 1.28    |


## A Hybrid Approach Based on Particle Swarm Optimization for Echo State Network Initialization

### Objective:
Pre-train the weights of an ESN using PSO

### Evolutionary algorithm:
- The ESN is initially trained using a linear regression approach, and the testing error is derived.
- Some randomly initialized weights are selected for optimization using PSO.
- After optimization, the updated weights are re-injected into the ESN.
- The ESN is re-trained, and the testing error is checked.

### Datasets:
Testing was done on the Mackey Glass dataset

| Reservoir Connectivity | Method   | Benchmark Delay  | MSE       
|------------------------|----------|------------------|-----------
| 0.1                    | ESN-PSO  | 17               | 8.09e-005 
| 0.1                    | ESN      | 17               | 4.96e-003 
| 0.1                    | ESN-PSO  | 30               | 9.50e-005 
| 0.1                    | ESN      | 30               | 2.87e-003 
| 0.25                   | ESN-PSO  | 17               | 8.12e-005 
| 0.25                   | ESN      | 17               | 5.82e-003 
| 0.25                   | ESN-PSO  | 30               | 9.61e-005 
| 0.25                   | ESN      | 30               | 8.97e-003 
| 0.4                    | ESN-PSO  | 17               | 9.06e-005 
| 0.4                    | ESN      | 17               | 7.82e-003 
| 0.4                    | ESN-PSO  | 30               | 9.73e-005 
| 0.4                    | ESN      | 30               | 1.90e-002 

## Comparing Evolutionary Methods for Reservoir Computing Pretraining

The objective of the study is to compare three evolutionary algorithms for finding the best reservoir in Reservoir Computing applied to time series forecasting in terms of prediction error and computational complexity. The three evolutionary approaches used in the study to find the best reservoir for time series forecasting are:
1. RCDESIGN (Reservoir Computing Design e Training):

2. Classical Searching: This approach uses Genetic Algorithms (GA) to search for the main RC generation parameters, which include reservoir size, spectral radius, and the density of interconnections of the W matrix.
3. Topology and Radius Searching

### Dataset:
This research used real data from the SONDA project to create a wind power model for energy planning in Brazil. Three wind speed time series were chosen for the experiments, obtained from wind headquarters in Triunfo, Belo Jardim, and São João do Cariri.

# ESN ensembling
## Adaptive Prognostic of Fuel Cells by Implementing Ensemble Echo State Networks in Time-Varying Model Space
This paper presents an adaptive data-driven prognostic strategy for fuel cells operated in different conditions. The strategy involves extracting a health indicator by identifying linear parameter varying models in sliding data segments and formulating virtual steady-state stack voltage in the identified model space. The paper proposes using an ensemble of multiple ESN models, each with different spectral radius and leakage rate parameters, to enhance the adaptability of the prognostic. The results of each ESN model are added together to get the final result. Long-term tests on a type of low power-scale proton exchange membrane fuel cell stack in different operating modes are carried out and the performance of the proposed strategy is evaluated using the experimental data. The paper concludes that the proposed strategy can provide an adaptive prediction of remaining useful life and its confidence interval even in dynamic conditions.


## Ensembles of Echo State Networks for Time Series Prediction
In this paper, ensemble techniques are employed to overcome the randomness and instability of echo state predictors, and a dynamic ensemble predictor is established. The proposed predictor is tested in numerical experiments and different strategies for training the predictor are also comparatively studied. A case study is then conducted to test the predictor’s performance in realistic prediction tasks.

The ensembling technique is to train a few ESNs with random reservoirs and stack them together by training an ELM model on their output. They tested two strategies with differing amount of training data for the ESN training and ELM training. The results of the two strategies on the Mackey glass dataset are as follows

| Metric                               | Strategy 1 | Strategy 2 |
| ------------------------------------ | ---------- | ---------- |
| Length of training set 1             | 250        | 90         |
| Length of training set 2             | 250        | 160        |
| Average testing error of components  | 2.11e-4    | 3.37e-4    |
| Training error of the ensemble       | 4.46e-7    | 1.15e-4    |
| Testing error of the ensemble        | 4.08e-4    | 1.51e-4    |


## Road Traffic Forecasting using Stacking Ensembles of Echo State Networks
this paper discusses road traffic forecasting using stacking ensembles of Echo State Networks. The authors explore how ensembles of Echo State Networks can yield improved traffic forecasts when compared to other machine learning models. They propose a regression model composed of a stacking ensemble of reservoir computing learners. The simulation results obtained with real data from Madrid, Spain show that the combination of stacking ensembles and reservoir computing allows their proposed model to outperform other machine learning models considered in their benchmark.

The models used in their experiments are as follows:
- A diversity of classical shallow learning models, including k-Nearest Neighbor Regression (kNNR), Linear Regression (LR), Decision Tree Regression (DTR), Extreme Learning Regression (ELR), Support Vector Regressor (SVR) and a Multilayer Perceptron (MLP).
- Three different bagging and boosting ensemble models: Random Forest (RFR), Adaboost (ADA) and Gradient Boosting (GBR). Hyper-parameter tuning is done for these models in a similar fashion to the previous set of models.
- A recurrent Deep Learning model (LSTM) comprising a single layer of LSTM neurons with hidden state size equal to 64, followed by a dense fully-connected layer and an ReLu output activation function. The model is trained for 100 epochs with mean squared error as the loss function to minimize, and Adam chosen as the solver for this purpose.
- Proposed stacking ensemble (s-ESN) consisting of M = 10 ESN models and a combiner that is also configured as another ESN learner. The leaking rate α, number of neurons in the repository N and activation function f(·) are set to 0.3, 100 and tanh for all ESN models

## Effective energy consumption forecasting using enhanced bagged echo state network