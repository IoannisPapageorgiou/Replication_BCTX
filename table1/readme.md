# Description

This folder folder contains data and code to reproduce the results of Table 1.

# Data

The folder "data" includes the datasets used in this experiment: the simulated datasets sim1, sim2, sim3, and the real-world datasets unemp, gnp, ibm. All real-world datasets contain the transformed (differenced, etc.) data, so they just need to be loaded in their exact form.

# BCT-AR code

The folder "bct_code" contains the C++ code for the BCT-AR model in the above foecasting experiments. For simplicity and ease of preproduction, a different ".cpp" file is given for each dataset, in which all the BCT-AR hyperparameters are already tuned appropriately for the corresponding experiments. More explicit instruction to compile the C++ code are given in the file "bct_code/help.md".  

# Code for alternative methods

The folder "alts" contains implementations of the alternative methods in R and Python. These files automatically load the data from the folder "table1/data", and compute the MSE forecasting results for the corresponding method. Specifically:

* The file 'forecast_pred.R' implements the methods ARIMA, ETS, and NNAR, using the R package 'forecast'.

* The file 'gluonts_pred.py' is Python code which implements the methods DeepAR and N-BEATS, using the library 'GluonTS'.

* The file 'msa_pred.R' implements the MSA model using the R package 'MSwM'.

* The file 'mar_pred.R' implements the MAR model using the R package 'mixAR'.

* The file 'tar_pred.R' implements the SETAR model using the R package 'TSA'.
