# Topology and parameter optimization

## Evolutionary strategy for simultaneous optimization of parameters, topology and reservoir weights in Echo State Networks 

This paper uses an evolutionary strategy in order to search the best values of the reservoir global parameters, the best topology and the reservoir weight sizes simultaneously. Listed below are the steps in the evolutionary algorithm
- Dataset creation: Time series is normalized and split into training/validation (75%) and testing (25%) sets.

- Population initialization: A random initial population of 120 individuals is created, each represented by a vector 's' with ESN information.

- Topology creation: For each 's', a topology is created with corresponding elements. All topologies have one input and one output.

- Readout training and performance evaluation: A readout function is trained, and reservoir performance is evaluated using 10-fold cross-validation.

- Fitness computation: Fitness value is calculated based on performance in training and validation sets.

- New populations: Algorithm creates new populations through elitism, crossover, and mutation operations.

- Stopping condition: Algorithm stops after 10 generations.

- Test performance: Best ESN performance is computed for the test set (25% of the data).

They used two wind speed time series from the SONDA project, collected at Triunfo and Belo Jardim in Brazil. The performance of their models can be summarized in the following tables:

| Dataset   | NMSE  | MSE(%) | MAPE(%) | MAE (m/s) |
|-----------|-------|--------|---------|-----------|
| Triunfo - Training   | 0.0803 | 0.1379 | 7.07    | 0.86    |
| Triunfo - Validation | 0.1697 | 0.1074 | 6.93    | 0.84    |
| Triunfo - Test       | 0.1886 | 0.1071 | 7.06    | 0.84    |
| Triunfo - Persistence| 0.7056 | 0.4006 | 14.43   | 1.70    |

| Dataset       | NMSE  | MSE(%) | MAPE(%) | MAE (m/s) |
|---------------|-------|--------|---------|-----------|
| Belo Jardim - Training   | 0.2544 | 0.5032 | 16.70   | 0.69    |
| Belo Jardim - Validation | 0.2751 | 0.4747 | 16.56   | 0.68    |
| Belo Jardim - Test       | 0.3016 | 0.5519 | 17.08   | 0.73    |
| Belo Jardim - Persistence| 0.8608 | 1.5749 | 30.16   | 1.28    |

The paper presents an approach to evolving all parameters of an ESN, however, the stopping condition of 10 generations might be too early to stop. This can be seen in the graph of lowest fitness score per generation which seems to be going down towards the end of the evolutionary algorithm. A stopping criteria based on performance gain might be more suitable. Additionally, the performance of the optimized ESNs is not compared with any baseline, making it difficult to ascertain the impact of the optimization process.


## A Hybrid Approach Based on Particle Swarm Optimization for Echo State Network Initialization

This paper is aimed at pre-training the weights of an ESN using PSO. Listed below are the steps in the algorithm used to pretrain the weights:
- The ESN is initially trained using a linear regression approach, and the testing error is derived.
- Some randomly initialized weights are selected for optimization using PSO.
- After optimization, the updated weights are re-injected into the ESN.
- The ESN is re-trained, and the testing error is checked.

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

The results from the experiments show that optimizing even a few weights of the reservoir can greatly improve the performance of ESNs. However, it is unclear from the paper what fraction of the weights were used in PSO optimization. Furthermore, the relationship between what fraction of the weights were optimized with PSO and the performance should be examined further.

## Comparing Evolutionary Methods for Reservoir Computing Pretraining

The objective of the study is to compare three evolutionary algorithms for finding the best reservoir in Reservoir Computing applied to time series forecasting in terms of prediction error and computational complexity. The three evolutionary approaches used in the study to find the best reservoir for time series forecasting are:
1. RCDESIGN (Reservoir Computing Design e Training): RCDESIGN uses GA and simultaneously looks for the best values of parameters, topology and weight matrices. The paper describes how an ESN can be encoded as an individual in GA. The crossover is an adaptation of uniform crossover. A mask is created that dictates which gene is inherited from which parent. Mutation is applied to every gene after the first, since the first gene defines the size

2. Classical Searching: Classical Searching uses GA in order to search the principal RC generation parameters, which are the reservoir size, the spectral radius, and the density of interconnections of W.

3. Topology and Radius Searching: This approach adds the search for optional connections such as feedback and bias to classical searching

This research used real data from the SONDA project to create a wind power model for energy planning in Brazil. Three wind speed time series were chosen for the experiments, obtained from wind headquarters in Triunfo, Belo Jardim, and São João do Cariri.
The performance of the models on all datasets went in the order RCDESIGN, Topology and Radius Searching, and classical searching, with RCDESIGN far exceeding the performance of classical searching

This paper shows that optimization of all hyper and meta parameters of ESNs is feasible and yields significantly better performance as compared to optimizing only hyperparameters. Additionally, they also present an approach that allows evolving reservoirs of different sizes while still including crossover between different individuals

## Single- and Multi-Objective Particle Swarm Optimization of Reservoir Structure in Echo State Network

This paper discusses the optimization of reservoir structure in Echo State Networks (ESNs) using single- and multi-objective particle swarm optimization (PSO). During the search process, an ESN is represented using the input, reservoir and feedback connectivity rates, and the reservoir size. In Single-objective PSO, the objective is the network error (RMSE/MSE). In bi-objective PSO, the objectives are minimization of MSE and connectivity rate. For tri-objective PSO, the objectives are minimization of MSE, reservoir connectivity rate and reservoir size

The proposed approaches are tested on various benchmarks, such as NARMA and Lorenz time series, and show good performance compared to other methods.

| Method                  | RMSE (Lorenz) | MSE (NARMA)|
| ----------------------- | ---------- | -----------|
| SOPSO-ESN              | 3.50 e-003 | 1.51 e-004 |
| MOPSO-ESN (2 objectives)| 3.24 e-002 | 3.96 e-004 |
| MOPSO-ESN (3 objectives)| 1.44 e-002 | 2.12 e-004 |

This paper demonstrates that optimizing ESNs using PSO can improve the performance, when compared to other approaches. However, using multi objective optimization has no benefit to performance. This could possibly be due to the selection of objectives, since minimizing reservoir size and connectivity are not conducive to higher performance but rather to reducing model complexity

## Growing Echo-State Network With Multiple Subreservoirs

This paper presents a growing Echo-State Network (ESN) with multiple subreservoirs. The proposed method, automatically designs the size and topology of the reservoir to match a given application. The GESN adds hidden units to the existing reservoir group by group, creating multiple subreservoirs. The algorithm starts with no subreservoirs, then, it generates a new subreservoir weight matrix using singular value decomposition. The input weight matrix is also randomly generated. The internal states of the growing reservoir are updated and collected, and the output weights are calculated using linear regression. The training error or validation error is calculated and if a stopping criterion is satisfied, the algorithm stops; otherwise, it continues to add more subreservoirs until the stopping criterion is met. GESN was evaluated on tenth order NARMA, the Mackey-glass dataset and the sunspot series. The lowest errors achieved on the datasets are as follows

| Dataset               | NRMSE|
| --------------------- | ---------- |
| NARMA-10              | 0.0843 |
| MG                    | 1.32 e-02 |
| MG (noisy)            | 3.14 e-02 |
| Sunspot               | 0.3416
| Sunspot (smoothed)    | 0.0254

The GESN approach is able to achieve competitve performance compared to other approaches, however, the major advantage of GESN is that it produces an ESN that has the least complexity needed to achieve good performance. This results in GESN producing much smaller models that can be trained faster and achieve competetive results.

## Evolutionary Echo State Network: Evolving Reservoirs in the Fourier Space **

This paper presents a new computational model called EvoESN (EVOLutionary Echo State Network) that represents the reservoir weights of an Echo State Network (ESN) in the Fourier space and performs fine-tuning of these weights using genetic algorithms in the frequency domain. The reasoning for using the Fourier space to represent ESN weights is that a direct encoding (that is, a one-to-one mapping where every weight is represented in the gene) is unsuitable for large-scale networks, furthermore, naive crossover operations present deficiencies when used for combining recurrent networks. The reservoir weights are projected into the frequency domain using Discrete Cosine Transform. This approach allows for a dimensionality reduction transformation of the initial method, and enables the exploitation of the benefits of large recurrent structures while avoiding training problems associated with gradient-based methods. EvoESN was evaluated on the Mackey glass system, Lorenz attractor and Monthly sunspot series. Only in the case of the sunspot dataset, EvoESN was able to achieve a distinctly good performance. In the case of the first two datasets, the performance was quite close to that in H. Jaeger et al. (add this paper to review later). An anomaly in the results is that the ESN with feedback model in H. Jaeger et al achieves a much lower performance than the ESN with feedback model trained by the authors of this paper


<br></br>
# ESN ensembling
## Adaptive Prognostic of Fuel Cells by Implementing Ensemble Echo State Networks in Time-Varying Model Space
This paper presents an adaptive data-driven prognostic strategy for fuel cells operated in different conditions. The strategy involves extracting a health indicator by identifying linear parameter varying models in sliding data segments and formulating virtual steady-state stack voltage in the identified model space. The paper proposes using an ensemble of multiple ESN models, each with different spectral radius and leakage rate parameters, to enhance the adaptability of the prognostic. The results of each ESN model are added together to get the final result. Long-term tests on a type of low power-scale proton exchange membrane fuel cell stack in different operating modes are carried out and the performance of the proposed strategy is evaluated using the experimental data. The paper concludes that the proposed strategy can provide an adaptive prediction of remaining useful life and its confidence interval even in dynamic conditions.
Selecting hyperparameters at random should result in model diversity, however, it can also result in models of low quality if the hyperparameter range is not analyzed beforehand. This can result in the proposed ensemble performing poorly since the ensemble simply takes the average of all predictions. Furthermore, the paper doesn't compare the performance of the ESN ensemble with an ESN, thus it is hard to determine how beneficial the proposed approach is over using a single ESN.


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

This paper provides insight into how training data should be used when stacking ESN models. However, one potential limitation is the method used for inducing model diversity. Unlike the previous paper that modified ESN parameters to create diverse models, this paper only relies on random weight initialization. Further research needs to be conducted on how diversity is best induced in ESN models.


## Road Traffic Forecasting using Stacking Ensembles of Echo State Networks
this paper discusses road traffic forecasting using stacking ensembles of Echo State Networks. The authors explore how ensembles of Echo State Networks can yield improved traffic forecasts when compared to other machine learning models. They propose a regression model composed of a stacking ensemble of reservoir computing learners and compare its performance with a few other types of models:

The models used in their experiments are as follows:
- A diversity of classical shallow learning models, including k-Nearest Neighbor Regression (kNNR), Linear Regression (LR), Decision Tree Regression (DTR), Extreme Learning Regression (ELR), Support Vector Regressor (SVR) and a Multilayer Perceptron (MLP).
- Three different bagging and boosting ensemble models: Random Forest (RFR), Adaboost (ADA) and Gradient Boosting (GBR). Hyper-parameter tuning is done for these models in a similar fashion to the previous set of models.
- A recurrent Deep Learning model (LSTM) comprising a single layer of LSTM neurons with hidden state size equal to 64, followed by a dense fully-connected layer and an ReLu output activation function. The model is trained for 100 epochs with mean squared error as the loss function to minimize, and Adam chosen as the solver for this purpose.
- Proposed stacking ensemble (s-ESN) consisting of M = 10 ESN models and a combiner that is also configured as another ESN learner. The leaking rate α, number of neurons in the repository N and activation function f(·) are set to 0.3, 100 and tanh for all ESN models

The simulation results obtained with real data from Madrid, Spain show that the combination of stacking ensembles and reservoir computing allows their proposed model to outperform other machine learning models considered in their benchmark.

## Effective energy consumption forecasting using enhanced bagged echo state network **

The study proposes a new enhanced optimization model based on the bagged echo state network improved by differential evolution algorithm to estimate energy consumption. The proposed model combines the merits of three techniques which are echo state network, bagging, and differential evolution algorithm. DE is a population-based algorithm that uses mutation, crossover, and selection operations to evolve a population of candidate solutions towards an optimal solution. DE is particularly effective for solving problems with continuous variables. In this research, DE is used to optimize the reservoir size, connectivity and spectral radius. Initially, Multiple repetitions of DE are used to evolve ESNs that are then trained and ensembled using bagging. The proposed model is applied to two comparative examples and an extended application to verify its accuracy and reliability. Results of the comparative examples show that the proposed model achieves better forecasting performance compared with basic echo state network and other existing popular models.
This paper follows a similar approach to ours, however, one limitation is that DE has to be conducted multiple times based on how many models the bagging requires. Multi-objective evolutionary algorithms can remove the need to carry out DE multiple times while also improving model diversity. The results of the paper do also show that using ensembling in combination with evolutionary algorithms leads to a better performance than using only one of the two.

## An ensemble quadratic echo state network for non-linear spatio-temporal forecasting

This paper presents an ensemble quadratic echo state network for non-linear spatio-temporal forecasting. A quadratic echo state network (QESN) is a modification of the basic echo state network (ESN) model that includes quadratic interactions between hidden processes and the response, as well as embeddings (lagged values) of the input. These modifications allow for more effective forecasts of non-linear spatio-temporal dynamical systems.
Testing was performed on Lorenz 40 and a real-world example for long-lead forecasting of tropical Pacific sea surface temperature (SST), showing promising results, however, it is difficult to comment on the results obtained in this paper since the performance of the ensembled QESN model was not compared with other models

<br></br>
# Research Gaps
- Unexplored objectives in evolutionary optimization (Memory capacity, separability, generalizability)
- Exploration of deep ESNs in evolutionary optimization
- In evolutionary algorithms, using crossover directly on the reservoir weights is not an effective approach (Safe mutations for deep and recurrent neural networks
through output gradients)
- Lack of analysis of ensembling techniques for optimizing ESN performance
- Lack of model diversity in ensemble learners (Need to see how to measure diversity and how it impacts model performance)
- Lack of performance data on commonly used benchmarks and comparison with baseline models
