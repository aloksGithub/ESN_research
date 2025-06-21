# Parameterizing echo state networks for multi-step time series prediction
> Reproduction package for: "Parameterizing echo state networks for multi-step time series prediction"
> https://doi.org/10.1016/j.neucom.2022.11.044
>
> In here we present a minimal example for the reproduction of the results in the above mentioned publication
>
> This code is forked from:
```
@online{ReservoirPy,
	Date = {08/09/2019},
	Title = {ReservoirPy},
	Url = {https://github.com/neuronalX/reservoirpy},
	}
```
```
Commit: c18f3b62bd788d79f1ead16a20684c6531c2540e
```
> current version: 
```
@incollection{Trouvain2020,
  doi = {10.1007/978-3-030-61616-8_40},
  url = {https://doi.org/10.1007/978-3-030-61616-8_40},
  year = {2020},
  publisher = {Springer International Publishing},
  pages = {494--505},
  author = {Nathan Trouvain and Luca Pedrelli and Thanh Trung Dinh and Xavier Hinaut},
  title = {{ReservoirPy}: An Efficient and User-Friendly Library to Design Echo State Networks},
  booktitle = {Artificial Neural Networks and Machine Learning {\textendash} {ICANN} 2020}
}



```
## Preparing Tests
### Installing Requirements
  - using pip3 to install required packages
    ```
    pip3 install -r req.txt
    ```
## Data
  - the generated datasets (Lorenz63, Mackey-Glass, Neutral) can be found as .npy files
  
  - Santa Fe laser is downloadable from: 
```
@online{Laser,
	Date = {08.12.2020 14:55},
	Title = {DynaML},
	Url = {https://github.com/transcendent-ai-labs/DynaML/tree/master/data},
	}
```
  - create a folder 'Samples' and move the datasets into it

## Run Tests
  - change dataset in script "example.py", predefined is the Mackey-Glass equation
  - uncomment parameter to optimize in the same script (Look for '# Select Hyper-Parameter')
  - save
  - run python script in terminal
    ```
    python3 example.py
    ``` 

---
**NOTE**

 - if only \beta is to be optimized and/or tested change 
    ```
    import ESN_Torch as ESN
    ```
    to
    ```
    import ESN_TorchBeta as ESN
    ```
    and call the calculation of Wout separately as described in script.

 - for additional tests regarding the R2-score the jupyter notebook 'R2_test.ipynb' is added
 - for additional tests regarding the length of the training interval 'RunForward.py' is added
 - the directories 'MG2021', 'Lorenz2021', 'Neutral2021' and 'Laser2021' have to be created
--- 
 
