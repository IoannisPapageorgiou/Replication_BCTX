The results of Table 3 replicate the forecasting experiment of Table 2 for a large number of major stock market indices, in order to test for statistical significance of the results. The complete list of the 21  stock market datasets that are used in this experiment are contained in the folder 'table3/data' below. Again, the appropriately transformed data are given, so these files can be loaded directly to the C++ and R code, respectively.

The code used to generate the results is identical with the corresponding code from Table 2; see folder 'table2' for details on these. The BCT-ARCH model is implemented in the C++ code 'bct_code/bctarch_pred.cpp', the GJR and EGARCH models are implemented in 'alts/all_garch_pred.R' using the R package rugarch, and the MSGARCH model in 'alts/msgarch_pred.R' using the package 'MSGARCH'. 


