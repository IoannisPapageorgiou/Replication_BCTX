# Description

This folder folder contains data and code to reproduce the results of Table 2.

# Data

The folder "data" includes the datasets used in this experiment, i.e., the (appropriately transformed) daily values of important stock market indices: FTSE 100, CAC 40, DAX and S&P 500.

# Code for the BCT-ARCH model

The folder "bct_code" contains the C++ code for the BCT-ARCH model in the above forecasting experiments. Since the same hyperparameters are selected for the BCT-ARCH model in all cases, a single "bctarch_pred.cpp" file is used here. The dataset is read in the main function: the first dataset 'ftse.txt' (from the subfolder "table2/data") is read here (corresponding to FTSE 100), but the other datasets can be loaded similarly. 

More specific instructions on how to compile the C++ code are identical with those for the BCT-AR model in the results of Table 1, which are given in the file "table1/bct_code/help.md".  

# Code for alternative methods

The folder "alts" contains implementations of the alternative methods in R. These files automatically load the data from the folder "table2/data", and compute the forecasting results for the corresponding method. Specifically:

* The file 'all_garch_pred.R' implements the methods ARCH, GARCH, GJR, and EGARCH using the R package 'rugarch'.

* The file 'msgarch_pred.R' implements the MSGARCH model using the R package 'MSGARCH'.

* The file 'stochvol_pred.R' implements the SV model using the R package 'stochvol'.



