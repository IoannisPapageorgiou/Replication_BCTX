# Description 

The results of Table 3 replicate the forecasting experiment of Table 2 for a large number of major stock market indices, in order to test for statistical significance of the results. This folder contains the data and code required to reproduce the results of Table 3.

# Data 
The complete list of the 21  stock market datasets that are used in this experiment are contained in the subfolder 'table3/data'. Again, the appropriately transformed data are given, so these files can be loaded directly to the C++ and R code.

# Code

The code used to generate the results is identical with the corresponding code from Table 2; see folder 'table2' for more details on these. The BCT-ARCH model is implemented in the C++ code 'bct_code/bctarch_pred.cpp'. The GJR and EGARCH models are implemented in 'alts/all_garch_pred.R' using the R package rugarch, and the MSGARCH model in 'alts/msgarch_pred.R' using the package 'MSGARCH'. 


